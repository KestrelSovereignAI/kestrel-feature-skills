const fs = require('node:fs/promises');
const path = require('node:path');

const { test, expect } = require('@playwright/test');

const API_KEY = process.env.KESTREL_API_KEY || '';
const AGENT = process.env.KESTREL_AGENT || 'kite';
const API_ROOT = `/api/agents/${encodeURIComponent(AGENT)}/api/procedural-skills`;
const SKILL_NAME = 'e2e-editor';
const DESCRIPTION = 'E2E editor sentinel';

function headers(extra = {}) {
  return {
    'X-API-Key': API_KEY,
    'X-Kestrel-Allow-Destructive': 'true',
    ...extra,
  };
}

async function catalogSkill(request, name) {
  const response = await request.get(API_ROOT, { headers: headers() });
  expect(response.ok(), await response.text()).toBeTruthy();
  return (await response.json()).skills.find((skill) => skill.name === name) || null;
}

async function deleteFixture(request, name) {
  const skill = await catalogSkill(request, name);
  if (!skill?.delete_revision) return null;
  return request.delete(`${API_ROOT}/${name}`, {
    headers: headers({ 'If-Match': skill.delete_revision }),
  });
}

async function patchState(request, name, data) {
  const skill = await catalogSkill(request, name);
  expect(skill, `expected ${name} in catalog before state mutation`).not.toBeNull();
  return request.patch(`${API_ROOT}/${name}/state`, {
    headers: headers({ 'If-Match': skill.revision }),
    data,
  });
}

async function resetFixture(request) {
  await deleteFixture(request, SKILL_NAME);
  const created = await request.post(API_ROOT, {
    headers: headers(),
    data: {
      name: SKILL_NAME,
      description: DESCRIPTION,
      body: '# Procedure\n\n1. Keep the editor round trip stable.',
      enabled: false,
    },
  });
  expect(created.ok(), await created.text()).toBeTruthy();
  const skill = await catalogSkill(request, SKILL_NAME);
  const script = await request.put(`${API_ROOT}/${SKILL_NAME}/file`, {
    headers: headers({ 'If-Match': skill.revision }),
    data: {
      path: 'scripts/risk.py',
      content: 'raise RuntimeError("must never execute from the editor")\n',
    },
  });
  expect(script.ok(), await script.text()).toBeTruthy();
}

async function openPanel(page) {
  await page.addInitScript((key) => {
    globalThis.sessionStorage.setItem('kestrel_api_key', key);
  }, API_KEY);
  await page.goto('/');
  await page.waitForLoadState('networkidle');
  const tab = page.locator('.nav-tab[data-panel="skills"]');
  if (!(await tab.isVisible().catch(() => false))) {
    const kite = page.getByText(new RegExp(`^${AGENT}$`, 'i')).last();
    if (await kite.isVisible().catch(() => false)) await kite.click();
  }
  if (!(await tab.isVisible().catch(() => false))) {
    const advanced = page.getByRole('button', { name: /Advanced/i });
    if (await advanced.isVisible().catch(() => false)) await advanced.click();
  }
  await expect(tab).toBeVisible({ timeout: 20_000 });
  await tab.click();
  await expect(tab).toHaveClass(/active/);
  await expect(page.locator('#panel-skills')).toBeVisible();
  await expect(page.locator('#panel-skills')).toContainText('Procedural skills');
  await expect.poll(
    () => page.evaluate(async () => {
      const module = await import('/js/api.js');
      return module.default.getHostAgent();
    }),
    { timeout: 20_000 },
  ).toBe(AGENT);
  await expect(page.locator('[role="status"]')).toContainText('Catalog loaded');
}

test.describe.serial('procedural skills contributed console', () => {
  test.skip(process.env.KESTREL_EXPECT_SKILLS === '0', 'feature-present suite');

  test.beforeAll(async ({ request }) => {
    expect(API_KEY, 'KESTREL_API_KEY is required for live console tests').not.toBe('');
    await resetFixture(request);
  });

  test.afterAll(async ({ request }) => {
    await deleteFixture(request, SKILL_NAME);
  });

  test('capability opt-out suppresses the contributed panel', async ({ page }) => {
    await page.addInitScript(({ key }) => {
      globalThis.sessionStorage.setItem('kestrel_api_key', key);
      globalThis.KESTREL_UI_CONFIG = {
        capabilities: { 'procedural-skills': false },
      };
    }, { key: API_KEY });
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await expect(page.locator('.nav-tab[data-panel="skills"]')).toHaveCount(0);
  });

  test('navigator renders the folder tree and opens SKILL.md', async ({ page }) => {
    await openPanel(page);
    await page.getByRole('button', { name: SKILL_NAME }).click();
    const scripts = page.getByRole('button', { name: /scripts/ });
    await expect(scripts).toHaveAttribute('aria-expanded', 'false');
    await scripts.click();
    await expect(scripts).toHaveAttribute('aria-expanded', 'true');
    await expect(page.getByRole('button', { name: /risk\.py/ })).toBeVisible();
    await expect(page.getByRole('button', { name: 'SKILL.md' })).toBeVisible();
    await page.getByRole('button', { name: 'SKILL.md' }).click();
    await expect(page.getByLabel('Skill file editor')).toHaveValue(new RegExp(DESCRIPTION));
    await expect(page.locator('#panel-skills')).toContainText('est. tokens');
  });

  test('admin add creates a disabled skill through the modal', async ({ page, request }) => {
    const name = 'e2e-added';
    await deleteFixture(request, name);
    try {
      await openPanel(page);
      await page.getByTestId('skills-add').click();
      const dialog = page.getByTestId('skills-create-dialog');
      await expect(dialog).toBeVisible();
      await dialog.getByLabel('New skill name').fill(name);
      await dialog.getByLabel('New skill description').fill('Added through the console');
      await dialog.getByLabel('New skill procedure').fill('# Procedure\n\n1. Stay disabled.');
      await dialog.getByRole('button', { name: 'Create disabled skill' }).click();
      await expect(page.locator('[role="status"]')).toContainText(`Created ${name}; it remains disabled.`);
      const catalog = await request.get(API_ROOT, { headers: headers() });
      const created = (await catalog.json()).skills.find((skill) => skill.name === name);
      expect(created.enabled).toBe(false);
    } finally {
      await deleteFixture(request, name);
    }
  });

  test('admin add rejects a trailing skill-name separator before HTTP', async ({ page, request }) => {
    const name = 'e2e-invalid-';
    await openPanel(page);
    await page.getByTestId('skills-add').click();
    const dialog = page.getByTestId('skills-create-dialog');
    const nameInput = dialog.getByLabel('New skill name');
    await nameInput.fill(name);
    await dialog.getByLabel('New skill description').fill('Must stay client-side');
    await dialog.getByLabel('New skill procedure').fill('# Procedure\n\n1. Refuse this name.');
    await dialog.getByRole('button', { name: 'Create disabled skill' }).click();

    await expect(dialog).toBeVisible();
    expect(await nameInput.evaluate((input) => input.validity.patternMismatch)).toBe(true);
    expect(await nameInput.evaluate((input) => input.validationMessage)).not.toBe('');
    const catalog = await request.get(API_ROOT, { headers: headers() });
    expect((await catalog.json()).skills.some((skill) => skill.name === name)).toBe(false);
  });

  test('Markdown save rejects invalid frontmatter and survives reload', async ({ page }) => {
    await openPanel(page);
    await page.getByRole('button', { name: SKILL_NAME }).click();
    await page.getByRole('button', { name: 'SKILL.md' }).click();
    const editor = page.getByLabel('Skill file editor');
    await expect(editor).toHaveValue(/^---/);
    const valid = await editor.inputValue();
    await editor.fill('not frontmatter\n');
    await page.getByTestId('skills-save').click();
    await expect(page.locator('[role="status"]')).toContainText('Save rejected:');
    await expect(page.locator('[role="status"]')).toContainText("must start with an exact '---'");
    await editor.fill(valid.replace('Keep the editor round trip stable.', 'Round trip changed and persisted.'));
    await page.getByTestId('skills-save').click();
    await expect(page.locator('[role="status"]')).toContainText('Saved');
    await page.reload();
    await openPanel(page);
    await page.getByRole('button', { name: SKILL_NAME }).click();
    await page.getByRole('button', { name: 'SKILL.md' }).click();
    await expect(page.getByLabel('Skill file editor')).toHaveValue(/Round trip changed and persisted\./);
  });

  test('stale file reads cannot overwrite the newly selected skill', async ({ page, request }) => {
    const first = 'e2e-race-first';
    const second = 'e2e-race-second';
    for (const [name, description] of [[first, 'First race sentinel'], [second, 'Second race sentinel']]) {
      await deleteFixture(request, name);
      const created = await request.post(API_ROOT, {
        headers: headers(),
        data: { name, description, body: `# Procedure\n\n${description}`, enabled: false },
      });
      expect(created.ok(), await created.text()).toBeTruthy();
    }
    let releaseRead;
    let markReadStarted;
    const readStarted = new Promise((resolve) => { markReadStarted = resolve; });
    const readRelease = new Promise((resolve) => { releaseRead = resolve; });
    await page.route(new RegExp(`${first}/file\\?path=SKILL.md$`), async (route) => {
      markReadStarted();
      await readRelease;
      await route.continue();
    });
    try {
      await openPanel(page);
      await page.getByRole('button', { name: first }).click();
      await page.getByRole('button', { name: 'SKILL.md' }).click();
      await readStarted;
      await page.getByRole('button', { name: second }).click();
      await page.getByRole('button', { name: 'SKILL.md' }).click();
      await expect(page.getByLabel('Skill file editor')).toHaveValue(/Second race sentinel/);
      releaseRead();
      await page.waitForTimeout(250);
      await expect(page.getByLabel('Skill file editor')).toHaveValue(/Second race sentinel/);
      await page.getByLabel('Skill file editor').fill(
        (await page.getByLabel('Skill file editor').inputValue()).replace(
          'Second race sentinel',
          'Second race edited safely',
        ),
      );
      await page.getByTestId('skills-save').click();
      await expect(page.locator('[role="status"]')).toContainText(`Saved ${second}/SKILL.md`);
      const firstFile = await request.get(`${API_ROOT}/${first}/file?path=SKILL.md`, { headers: headers() });
      const secondFile = await request.get(`${API_ROOT}/${second}/file?path=SKILL.md`, { headers: headers() });
      expect((await firstFile.json()).content).toContain('First race sentinel');
      expect((await secondFile.json()).content).toContain('Second race edited safely');
    } finally {
      releaseRead?.();
      await page.unrouteAll({ behavior: 'ignoreErrors' });
      await deleteFixture(request, first);
      await deleteFixture(request, second);
    }
  });

  test('agent switch invalidates unsaved editor ownership', async ({ page }) => {
    await openPanel(page);
    await page.getByRole('button', { name: SKILL_NAME }).click();
    await page.getByRole('button', { name: 'SKILL.md' }).click();
    const editor = page.getByLabel('Skill file editor');
    await expect(editor).toBeEnabled();
    await editor.fill(`${await editor.inputValue()}\nUNSAVED-CROSS-AGENT-SENTINEL\n`);

    await page.evaluate(async () => {
      const bus = (await import('/js/ui-ext/bus.js')).default;
      bus.emit('agent:switch', { prev: 'kite', next: 'other-agent' });
    });

    await expect(page.locator('.nav-tab[data-panel="skills"]')).toHaveCount(0);
    await expect(editor).toHaveCount(0);
    await expect(page.getByTestId('skills-save')).toHaveCount(0);
  });

  test('route disappearance removes the stale panel and recovery restores it', async ({ page }) => {
    await openPanel(page);
    const tab = page.locator('.nav-tab[data-panel="skills"]');
    await page.route(new RegExp(`${API_ROOT}$`), async (route) => {
      await route.fulfill({
        status: 404,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'Feature route unavailable' }),
      });
    });

    await page.evaluate(() => {
      globalThis.dispatchEvent(new CustomEvent('capabilities:changed'));
    });
    await expect(tab).toHaveCount(0);

    await page.unrouteAll({ behavior: 'wait' });
    await page.evaluate(() => {
      globalThis.dispatchEvent(new CustomEvent('capabilities:changed'));
    });
    await expect(tab).toBeVisible();
    await tab.click();
    await expect(tab).toHaveClass(/active/);
    await expect(page.locator('#panel-skills')).toBeVisible();
    await expect(page.getByRole('button', { name: SKILL_NAME })).toBeVisible();
  });

  test('superseding catalog failure still tears down unavailable feature UI', async ({ page }) => {
    await openPanel(page);
    const tab = page.locator('.nav-tab[data-panel="skills"]');
    let requestCount = 0;
    let releaseFirst;
    const firstReleased = new Promise((resolve) => { releaseFirst = resolve; });
    let markFirstStarted;
    const firstStarted = new Promise((resolve) => { markFirstStarted = resolve; });
    await page.route(new RegExp(`${API_ROOT}(?:/reload)?$`), async (route) => {
      requestCount += 1;
      if (requestCount === 1) {
        markFirstStarted();
        await firstReleased;
      }
      await route.fulfill({
        status: 404,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'Feature route unavailable' }),
      });
    });
    try {
      await page.evaluate(() => {
        globalThis.dispatchEvent(new CustomEvent('capabilities:changed'));
      });
      await firstStarted;
      await page.getByTestId('skills-reload').click();
      await expect.poll(() => requestCount).toBeGreaterThanOrEqual(2);
      releaseFirst();
      await expect(tab).toHaveCount(0);
      await expect(page.locator('#panel-skills')).toHaveCount(0);
    } finally {
      releaseFirst();
      await page.unrouteAll({ behavior: 'ignoreErrors' });
    }
  });

  test('recreated active panel container remounts its feature body', async ({ page }) => {
    await openPanel(page);
    await page.evaluate(async () => {
      document.getElementById('panel-skills')?.remove();
      const panels = await import('/js/ui-ext/panels.js');
      panels.syncNav();
      document.querySelector('.nav-tab[data-panel="skills"]')?.click();
    });

    await expect(page.locator('#panel-skills')).toBeVisible();
    await expect(page.getByRole('button', { name: SKILL_NAME })).toBeVisible();
  });

  test('successful capability refresh preserves the active unsaved editor', async ({ page }) => {
    await openPanel(page);
    await page.getByRole('button', { name: SKILL_NAME }).click();
    await page.getByRole('button', { name: 'SKILL.md' }).click();
    const editor = page.getByLabel('Skill file editor');
    const unsaved = `${await editor.inputValue()}\nUNSAVED-CAPABILITY-SENTINEL\n`;
    await editor.fill(unsaved);

    await page.evaluate(() => {
      globalThis.dispatchEvent(new CustomEvent('capabilities:changed'));
    });

    await expect(editor).toBeVisible();
    await expect(editor).toHaveValue(unsaved);
    await expect(page.locator('#panel-skills')).toBeVisible();
  });

  test('catalog refresh updates selected state controls without losing the editor', async ({ page, request }) => {
    await patchState(request, SKILL_NAME, { enabled: false, priority: 100 });
    await openPanel(page);
    await page.getByRole('button', { name: SKILL_NAME }).click();
    await page.getByRole('button', { name: 'SKILL.md' }).click();
    const editor = page.getByLabel('Skill file editor');
    const unsaved = `${await editor.inputValue()}\nUNSAVED-REFRESHED-STATE-SENTINEL\n`;
    await editor.fill(unsaved);
    try {
      const externalUpdate = await patchState(
        request,
        SKILL_NAME,
        { enabled: true, priority: 23 },
      );
      expect(externalUpdate.ok(), await externalUpdate.text()).toBeTruthy();
      await page.evaluate(() => {
        globalThis.dispatchEvent(new CustomEvent('capabilities:changed'));
      });

      await expect(page.getByLabel('Skill priority')).toHaveValue('23');
      await expect(page.getByRole('button', { name: 'Disable', exact: true })).toBeVisible();
      await expect(editor).toHaveValue(unsaved);

      await page.getByLabel('Skill priority').fill('37');
      await page.getByRole('button', { name: 'Set priority' }).click();
      await expect(page.locator('[role="status"]')).toContainText(`Updated ${SKILL_NAME} priority to 37.`);
      const catalog = await request.get(API_ROOT, { headers: headers() });
      const skill = (await catalog.json()).skills.find((item) => item.name === SKILL_NAME);
      expect(skill.enabled).toBe(true);
      expect(skill.priority).toBe(37);
    } finally {
      await patchState(request, SKILL_NAME, { enabled: false, priority: 100 });
    }
  });

  test('state controls stay serialized while an update is in flight', async ({ page, request }) => {
    await patchState(request, SKILL_NAME, { enabled: false, priority: 100 });
    await openPanel(page);
    await page.getByRole('button', { name: SKILL_NAME }).click();
    let releasePatch;
    const patchReleased = new Promise((resolve) => { releasePatch = resolve; });
    let markCommitted;
    const patchCommitted = new Promise((resolve) => { markCommitted = resolve; });
    await page.route(new RegExp(`${API_ROOT}/${SKILL_NAME}/state$`), async (route) => {
      const response = await route.fetch();
      markCommitted();
      await patchReleased;
      await route.fulfill({ response });
    });
    try {
      await page.getByRole('button', { name: 'Enable', exact: true }).click();
      await patchCommitted;
      await expect(page.getByLabel('Skill priority')).toBeDisabled();
      await expect(page.getByRole('button', { name: 'Set priority' })).toBeDisabled();
      releasePatch();
      await expect(page.locator('[role="status"]')).toContainText(`Enabled ${SKILL_NAME}`);
      await expect(page.getByLabel('Skill priority')).toBeEnabled();
      await page.getByLabel('Skill priority').fill('37');
      await page.getByRole('button', { name: 'Set priority' }).click();
      await expect(page.locator('[role="status"]')).toContainText(`Updated ${SKILL_NAME} priority to 37.`);
      const catalog = await request.get(API_ROOT, { headers: headers() });
      const skill = (await catalog.json()).skills.find((item) => item.name === SKILL_NAME);
      expect(skill.enabled).toBe(true);
      expect(skill.priority).toBe(37);
    } finally {
      releasePatch();
      await page.unrouteAll({ behavior: 'ignoreErrors' });
      await patchState(request, SKILL_NAME, { enabled: false, priority: 100 });
    }
  });

  test('state completion preserves a newer skill selection and unsaved edit', async ({ page, request }) => {
    const newer = 'e2e-state-race-newer';
    await deleteFixture(request, newer);
    await request.post(API_ROOT, {
      headers: headers(),
      data: {
        name: newer,
        description: 'Newer state-race selection',
        body: '# Procedure\n\nKeep this unsaved edit.',
        enabled: false,
      },
    });
    await patchState(request, SKILL_NAME, { enabled: false, priority: 100 });
    await openPanel(page);
    await page.getByRole('button', { name: SKILL_NAME }).click();
    let releasePatch;
    const patchReleased = new Promise((resolve) => { releasePatch = resolve; });
    let markCommitted;
    const patchCommitted = new Promise((resolve) => { markCommitted = resolve; });
    await page.route(new RegExp(`${API_ROOT}/${SKILL_NAME}/state$`), async (route) => {
      const response = await route.fetch();
      markCommitted();
      await patchReleased;
      await route.fulfill({ response });
    });
    try {
      await page.getByRole('button', { name: 'Enable', exact: true }).click();
      await patchCommitted;
      await page.getByRole('button', { name: newer }).click();
      await page.getByRole('button', { name: 'SKILL.md' }).click();
      const editor = page.getByLabel('Skill file editor');
      const unsaved = `${await editor.inputValue()}\nUNSAVED-STATE-RACE-SENTINEL\n`;
      await editor.fill(unsaved);
      releasePatch();
      await expect(page.locator('[role="status"]')).toContainText(`Enabled ${SKILL_NAME}`);
      await expect(page.getByRole('button', { name: newer })).toHaveAttribute('aria-current', 'true');
      await expect(editor).toHaveValue(unsaved);
    } finally {
      releasePatch();
      await page.unrouteAll({ behavior: 'ignoreErrors' });
      await patchState(request, SKILL_NAME, { enabled: false, priority: 100 });
      await deleteFixture(request, newer);
    }
  });

  test('privacy indicator transition clears a concealed persisted editor', async ({ page }) => {
    await openPanel(page);
    await page.getByRole('button', { name: SKILL_NAME }).click();
    await page.getByRole('button', { name: 'SKILL.md' }).click();
    const editor = page.getByLabel('Skill file editor');
    await expect(editor).toHaveValue(new RegExp(DESCRIPTION));
    await page.route(new RegExp(`${API_ROOT}$`), async (route) => {
      if (route.request().method() !== 'GET') return route.continue();
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          skills: [],
          errors: [],
          count: 0,
          context: { text: '', included: [], dropped: [], token_costs: {} },
        }),
      });
    });
    try {
      await page.evaluate(() => {
        const indicator = document.getElementById('chat-privacy-indicator')
          || document.getElementById('privacy-indicator');
        if (!indicator) throw new Error('privacy indicator is unavailable');
        indicator.replaceChildren(Object.assign(document.createElement('span'), {
          textContent: 'Ephemeral',
        }));
      });
      await expect(page.getByRole('button', { name: SKILL_NAME })).toHaveCount(0);
      await expect(editor).toHaveValue('');
      await expect(editor).toBeDisabled();
    } finally {
      await page.unrouteAll({ behavior: 'ignoreErrors' });
    }
  });

  test('Python editor exposes execution risk and has no run surface', async ({ page }) => {
    await openPanel(page);
    await page.getByRole('button', { name: SKILL_NAME }).click();
    await page.getByRole('button', { name: /scripts/ }).click();
    await page.getByRole('button', { name: /risk\.py/ }).click();
    await expect(page.getByTestId('python-execution-risk')).toBeVisible();
    await expect(page.getByTestId('python-execution-risk')).toContainText('no run control');
    await expect(page.getByRole('button', { name: /run/i })).toHaveCount(0);
  });

  test('admin enable and disable changes the context breakdown', async ({ page, request }) => {
    await openPanel(page);
    await page.getByRole('button', { name: SKILL_NAME }).click();
    await page.getByRole('button', { name: 'Enable', exact: true }).click();
    await expect(page.locator('[role="status"]')).toContainText(`Enabled ${SKILL_NAME}`);
    let catalog = await request.get(API_ROOT, { headers: headers() });
    let payload = await catalog.json();
    expect(payload.context.included).toContain(SKILL_NAME);
    expect(payload.context.text).toContain(DESCRIPTION);
    await page.getByLabel('Skill priority').fill('11');
    await page.getByRole('button', { name: 'Set priority' }).click();
    await expect(page.locator('[role="status"]')).toContainText(`Updated ${SKILL_NAME} priority to 11.`);
    catalog = await request.get(API_ROOT, { headers: headers() });
    payload = await catalog.json();
    expect(payload.skills.find((skill) => skill.name === SKILL_NAME).priority).toBe(11);
    await page.getByRole('button', { name: SKILL_NAME }).click();
    await page.getByRole('button', { name: 'Disable', exact: true }).click();
    await expect(page.locator('[role="status"]')).toContainText(`Disabled ${SKILL_NAME}`);
    catalog = await request.get(API_ROOT, { headers: headers() });
    payload = await catalog.json();
    expect(payload.context.included).not.toContain(SKILL_NAME);
  });

  test('blank priority is rejected without changing persisted state', async ({ page, request }) => {
    await patchState(request, SKILL_NAME, { enabled: false, priority: 29 });
    await openPanel(page);
    await page.getByRole('button', { name: SKILL_NAME }).click();
    await page.getByLabel('Skill priority').fill('');
    await page.getByRole('button', { name: 'Set priority' }).click();
    await expect(page.locator('[role="status"]')).toContainText('Priority is required');
    const skill = await catalogSkill(request, SKILL_NAME);
    expect(skill.priority).toBe(29);
  });

  test('discover reload finds a new folder without restarting', async ({ page }) => {
    const root = process.env.KESTREL_KITE_SKILLS_ROOT;
    test.skip(!root, 'KESTREL_KITE_SKILLS_ROOT is required for filesystem discovery');
    const name = 'e2e-discovered';
    const folder = path.join(root, name);
    await fs.rm(folder, { recursive: true, force: true });
    await fs.mkdir(folder, { recursive: false });
    await fs.writeFile(
      path.join(folder, 'SKILL.md'),
      '---\nname: "e2e-discovered"\ndescription: "Discovered without restart"\n---\n\nProcedure.\n',
      'utf8',
    );
    try {
      await openPanel(page);
      await expect(page.getByRole('button', { name })).toHaveCount(0);
      await page.getByTestId('skills-reload').click();
      await expect(page.locator('[role="status"]')).toContainText('without restarting');
      await expect(page.getByRole('button', { name })).toBeVisible();
    } finally {
      await fs.rm(folder, { recursive: true, force: true });
    }
  });

  test('discover reload refreshes the selected resource tree', async ({ page }) => {
    const root = process.env.KESTREL_KITE_SKILLS_ROOT;
    test.skip(!root, 'KESTREL_KITE_SKILLS_ROOT is required for filesystem discovery');
    const resource = path.join(root, SKILL_NAME, 'outside-reload.md');
    await fs.rm(resource, { force: true });
    try {
      await openPanel(page);
      await page.getByRole('button', { name: SKILL_NAME }).click();
      await page.getByRole('button', { name: 'SKILL.md' }).click();
      const editor = page.getByLabel('Skill file editor');
      const unsaved = `${await editor.inputValue()}\nUNSAVED-RELOAD-SENTINEL\n`;
      await editor.fill(unsaved);
      await expect(page.getByRole('button', { name: 'outside-reload.md' })).toHaveCount(0);
      await fs.writeFile(resource, 'Added outside the Console.\n', 'utf8');
      await page.getByTestId('skills-reload').click();
      await expect(page.locator('[role="status"]')).toContainText('without restarting');
      await expect(page.getByRole('button', { name: 'outside-reload.md' })).toBeVisible();
      await expect(editor).toHaveValue(unsaved);
      await fs.rm(resource, { force: true });
      await page.getByTestId('skills-reload').click();
      await expect(page.getByRole('button', { name: 'outside-reload.md' })).toHaveCount(0);
      await expect(editor).toHaveValue(unsaved);
    } finally {
      await fs.rm(resource, { force: true });
    }
  });

  test('stale delete approval cannot remove a same-named replacement', async ({ page, request }) => {
    await openPanel(page);
    await page.getByRole('button', { name: SKILL_NAME }).click();
    await page.getByRole('button', { name: 'Delete', exact: true }).click();
    const dialog = page.getByTestId('skills-delete-approval');
    await expect(dialog).toBeVisible();
    await deleteFixture(request, SKILL_NAME);
    const replacement = await request.post(API_ROOT, {
      headers: headers(),
      data: {
        name: SKILL_NAME,
        description: 'Replacement must survive stale approval',
        body: '# Procedure\n\nReplacement.',
        enabled: false,
      },
    });
    expect(replacement.ok(), await replacement.text()).toBeTruthy();
    try {
      await dialog.getByTestId('skills-delete-confirm').click();
      await expect(dialog).not.toBeVisible();
      await expect(page.locator('[role="status"]')).toContainText('changed before deletion');
      const preserved = await catalogSkill(request, SKILL_NAME);
      expect(preserved.description).toBe('Replacement must survive stale approval');
    } finally {
      await resetFixture(request);
    }
  });

  test('partial cleanup is visibly reported as an error', async ({ page }) => {
    await openPanel(page);
    await page.getByRole('button', { name: SKILL_NAME }).click();
    await page.route(new RegExp(`${API_ROOT}/${SKILL_NAME}$`), async (route) => {
      if (route.request().method() !== 'DELETE') return route.continue();
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          name: SKILL_NAME,
          removed_file: true,
          config_deleted: false,
          graph_deleted: false,
          errors: ['enablement row cleanup failed', 'graph index cleanup failed'],
          resolved_skill_retained: false,
          remaining_source_kind: null,
        }),
      });
    });
    try {
      await page.getByRole('button', { name: 'Delete', exact: true }).click();
      await page.getByTestId('skills-delete-confirm').click();
      await expect(page.locator('[role="status"]')).toContainText('cleanup is incomplete');
      await expect(page.locator('[role="status"]')).toContainText('graph index cleanup failed');
      await expect(page.getByRole('button', { name: SKILL_NAME })).toBeVisible();
    } finally {
      await page.unrouteAll({ behavior: 'ignoreErrors' });
    }
  });

  test('delete requires the destructive confirmation dialog', async ({ page, request }) => {
    await openPanel(page);
    await page.getByRole('button', { name: SKILL_NAME }).click();
    await page.getByRole('button', { name: 'Delete', exact: true }).click();
    const dialog = page.getByTestId('skills-delete-approval');
    await expect(dialog).toBeVisible();
    let catalog = await request.get(API_ROOT, { headers: headers() });
    expect((await catalog.json()).skills.map((skill) => skill.name)).toContain(SKILL_NAME);
    await dialog.getByRole('button', { name: 'Cancel' }).click();
    await expect(dialog).not.toBeVisible();
    catalog = await request.get(API_ROOT, { headers: headers() });
    expect((await catalog.json()).skills.map((skill) => skill.name)).toContain(SKILL_NAME);
    await page.getByRole('button', { name: 'Delete', exact: true }).click();
    await dialog.getByTestId('skills-delete-confirm').click();
    await expect(page.locator('[role="status"]')).toContainText(`Deleted ${SKILL_NAME}`);
    catalog = await request.get(API_ROOT, { headers: headers() });
    expect((await catalog.json()).skills.map((skill) => skill.name)).not.toContain(SKILL_NAME);
  });
});
