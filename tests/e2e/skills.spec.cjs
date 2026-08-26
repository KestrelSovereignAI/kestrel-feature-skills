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

async function resetFixture(request) {
  await request.delete(`${API_ROOT}/${SKILL_NAME}`, { headers: headers() });
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
  const script = await request.put(`${API_ROOT}/${SKILL_NAME}/file`, {
    headers: headers(),
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
  const tab = page.locator('.nav-tab[data-panel="procedural-skills"]');
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
  await expect(page.locator('#panel-procedural-skills')).toBeVisible();
  await expect(page.locator('#panel-procedural-skills')).toContainText('Procedural skills');
  await expect(page.locator('[role="status"]')).toContainText('Catalog loaded');
}

test.describe.serial('procedural skills contributed console', () => {
  test.skip(process.env.KESTREL_EXPECT_SKILLS === '0', 'feature-present suite');

  test.beforeAll(async ({ request }) => {
    expect(API_KEY, 'KESTREL_API_KEY is required for live console tests').not.toBe('');
    await resetFixture(request);
  });

  test.afterAll(async ({ request }) => {
    await request.delete(`${API_ROOT}/${SKILL_NAME}`, { headers: headers() });
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
    await expect(page.locator('#panel-procedural-skills')).toContainText('est. tokens');
  });

  test('admin add creates a disabled skill through the modal', async ({ page, request }) => {
    const name = 'e2e-added';
    await request.delete(`${API_ROOT}/${name}`, { headers: headers() });
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
      await request.delete(`${API_ROOT}/${name}`, { headers: headers() });
    }
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
