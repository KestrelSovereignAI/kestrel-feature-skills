import API from '/js/api.js';
import { registerPanel, syncNav } from '/js/ui-ext/panels.js';
import bus from '/js/ui-ext/bus.js';

const PANEL_ID = 'procedural-skills';
const ROOT = '/api/procedural-skills';
const state = {
  active: false,
  catalog: null,
  selected: null,
  path: null,
  editorOwner: null,
  selectionEpoch: 0,
  fileEpoch: 0,
  catalogEpoch: 0,
  availabilityEpoch: 0,
  available: true,
  ui: null,
};

function currentAgent() {
  return typeof API.getHostAgent === 'function' ? API.getHostAgent() : null;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function button(label, handler, className = 'skills-button') {
  const node = el('button', className, label);
  node.type = 'button';
  node.addEventListener('click', handler);
  return node;
}

async function request(path = '', options = {}) {
  return API.request(`${ROOT}${path}`, options);
}

function detail(error) {
  return error?.detail || error?.message || String(error || 'Unknown skill error');
}

function setStatus(message, isError = false) {
  if (!state.ui) return;
  state.ui.status.textContent = message || '';
  state.ui.status.style.color = isError ? 'var(--error-color, #c0392b)' : '';
}

function mount(container) {
  const toolbar = el('div', 'skills-toolbar');
  const reload = button('Discover / reload', reloadCatalog);
  reload.dataset.testid = 'skills-reload';
  const add = button('Add skill', showCreateDialog);
  add.dataset.testid = 'skills-add';
  toolbar.append(reload, add);

  const list = el('ul', 'skills-list');
  const errors = el('ul', 'skills-errors');
  const left = el('aside', 'skills-card');
  left.append(el('h3', '', 'Catalog'), list, errors);

  const title = el('h3', '', 'Select a skill');
  const meta = el('div', 'skills-muted');
  const controls = el('div', 'skills-row');
  const tree = el('ul', 'skills-tree');
  const warning = el('div', 'skills-risk');
  warning.dataset.testid = 'python-execution-risk';
  warning.textContent = 'Execution risk: Python here is third-party code. This editor can save text only and has no run control.';
  warning.hidden = true;
  const editor = el('textarea', 'skills-editor');
  editor.spellcheck = false;
  editor.disabled = true;
  editor.setAttribute('aria-label', 'Skill file editor');
  const save = button('Save file', saveFile);
  save.disabled = true;
  save.dataset.testid = 'skills-save';
  const editorActions = el('div', 'skills-editor-actions');
  editorActions.append(save);
  const right = el('section', 'skills-card');
  right.append(title, meta, controls, tree, warning, editor, editorActions);

  const layout = el('div', 'skills-layout');
  layout.append(left, right);
  const status = el('div', 'skills-status');
  status.setAttribute('role', 'status');

  const createDialog = buildCreateDialog();
  const deleteDialog = buildDeleteDialog();
  container.replaceChildren(
    el('h2', '', 'Procedural skills'),
    el('p', 'skills-muted', 'Descriptions enter context only when enabled. Procedures and bundled files are disclosed explicitly.'),
    toolbar,
    layout,
    status,
    createDialog,
    deleteDialog,
  );
  container.classList.add('skills-panel');
  state.ui = { list, errors, title, meta, controls, tree, warning, editor, save, status, createDialog, deleteDialog };
  state.catalog = null;
  clearSelection();
  loadCatalog();
}

function buildCreateDialog() {
  const dialog = el('dialog', 'skills-dialog');
  dialog.dataset.testid = 'skills-create-dialog';
  const form = el('form');
  form.method = 'dialog';
  const name = el('input');
  name.name = 'name';
  name.required = true;
  name.pattern = '[a-z0-9][a-z0-9_-]{0,63}';
  name.placeholder = 'skill-name';
  name.setAttribute('aria-label', 'New skill name');
  const description = el('input');
  description.name = 'description';
  description.required = true;
  description.placeholder = 'One-line trigger description';
  description.setAttribute('aria-label', 'New skill description');
  const body = el('textarea', 'skills-editor');
  body.name = 'body';
  body.required = true;
  body.placeholder = '# Procedure\n\n1. ...';
  body.setAttribute('aria-label', 'New skill procedure');
  const cancel = button('Cancel', () => dialog.close());
  const submit = el('button', 'skills-button', 'Create disabled skill');
  submit.type = 'submit';
  const actions = el('div', 'skills-editor-actions');
  actions.append(cancel, submit);
  form.append(el('h3', '', 'Add procedural skill'), name, description, body, actions);
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const agent = currentAgent();
    try {
      const createdName = name.value;
      await request('', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name.value, description: description.value, body: body.value, enabled: false }),
      });
      if (currentAgent() !== agent) return;
      dialog.close();
      form.reset();
      await loadCatalog();
      if (currentAgent() === agent) {
        setStatus(`Created ${createdName}; it remains disabled.`);
      }
    } catch (error) {
      if (currentAgent() === agent) setStatus(detail(error), true);
    }
  });
  dialog.append(form);
  return dialog;
}

function buildDeleteDialog() {
  const dialog = el('dialog', 'skills-dialog');
  dialog.dataset.testid = 'skills-delete-approval';
  const message = el('p', '', 'This permanently deletes authored work. It cannot be undone.');
  const cancel = button('Cancel', () => dialog.close('cancel'));
  const approve = button('Delete permanently', async () => {
    const name = dialog.dataset.skill;
    if (!name) return;
    const agent = currentAgent();
    try {
      await request(`/${encodeURIComponent(name)}`, {
        method: 'DELETE',
        headers: { 'X-Kestrel-Allow-Destructive': 'operator-confirmed-ui' },
      });
      if (currentAgent() !== agent) return;
      dialog.close('approved');
      clearSelection();
      await loadCatalog();
      if (currentAgent() === agent) setStatus(`Deleted ${name}.`);
    } catch (error) {
      if (currentAgent() === agent) setStatus(detail(error), true);
    }
  }, 'skills-button skills-danger');
  approve.dataset.testid = 'skills-delete-confirm';
  const actions = el('div', 'skills-editor-actions');
  actions.append(cancel, approve);
  dialog.append(el('h3', '', 'Approve destructive deletion?'), message, actions);
  return dialog;
}

function showCreateDialog() {
  state.ui?.createDialog.showModal();
}

async function loadCatalog() {
  if (!state.ui) return;
  const agent = currentAgent();
  const epoch = ++state.catalogEpoch;
  try {
    const catalog = await request('');
    if (epoch !== state.catalogEpoch || currentAgent() !== agent) return;
    state.catalog = catalog;
    renderCatalog();
    setStatus(
      state.catalog.errors?.length || state.catalog.enablement_error
        ? 'Some skill state could not be loaded; see catalog errors.'
        : 'Catalog loaded.',
      Boolean(state.catalog.enablement_error),
    );
  } catch (error) {
    if (epoch === state.catalogEpoch && currentAgent() === agent) {
      setStatus(detail(error), true);
    }
  }
}

async function reloadCatalog() {
  const agent = currentAgent();
  const epoch = ++state.catalogEpoch;
  try {
    const catalog = await request('/reload', { method: 'POST' });
    if (epoch !== state.catalogEpoch || currentAgent() !== agent) return;
    state.catalog = catalog;
    renderCatalog();
    setStatus('Discovered skill folders without restarting.');
  } catch (error) {
    if (epoch === state.catalogEpoch && currentAgent() === agent) {
      setStatus(detail(error), true);
    }
  }
}

function renderCatalog() {
  const ui = state.ui;
  if (!ui) return;
  ui.list.replaceChildren();
  ui.errors.replaceChildren();
  for (const skill of state.catalog?.skills || []) {
    const row = el('li');
    const select = button(`${skill.name}${skill.enabled ? ' • enabled' : ''}`, () => selectSkill(skill.name));
    select.setAttribute('aria-current', String(state.selected === skill.name));
    row.append(select, el('div', 'skills-description', `${skill.token_cost} est. tokens · ${skill.source_id} · priority ${skill.priority}`));
    ui.list.append(row);
  }
  for (const error of state.catalog?.errors || []) {
    ui.errors.append(el('li', 'skills-risk', `${error.source_id}/${error.locator}: ${error.error}`));
  }
  if (state.catalog?.enablement_error) {
    ui.errors.append(el('li', 'skills-risk', `Enablement database: ${state.catalog.enablement_error}`));
  }
  if (!state.catalog?.skills?.length) ui.list.append(el('li', 'skills-muted', 'No valid skill folders discovered.'));
  if (state.selected && !(state.catalog?.skills || []).some((item) => item.name === state.selected)) clearSelection();
}

function clearSelection() {
  state.selectionEpoch += 1;
  state.fileEpoch += 1;
  state.selected = null;
  state.path = null;
  state.editorOwner = null;
  const ui = state.ui;
  if (!ui) return;
  ui.title.textContent = 'Select a skill';
  ui.meta.textContent = '';
  ui.controls.replaceChildren();
  ui.tree.replaceChildren();
  ui.editor.value = '';
  ui.editor.disabled = true;
  ui.save.disabled = true;
  ui.warning.hidden = true;
}

async function selectSkill(name) {
  const agent = currentAgent();
  const epoch = ++state.selectionEpoch;
  state.fileEpoch += 1;
  state.selected = name;
  state.path = null;
  state.editorOwner = null;
  if (state.ui) {
    state.ui.editor.value = '';
    state.ui.editor.disabled = true;
    state.ui.save.disabled = true;
    state.ui.warning.hidden = true;
  }
  renderCatalog();
  const skill = (state.catalog?.skills || []).find((item) => item.name === name);
  if (!skill || !state.ui) return;
  state.ui.title.textContent = skill.name;
  state.ui.meta.textContent = `${skill.description} · ${skill.source_kind} · ${skill.token_cost} estimated tokens`;
  const toggle = button(skill.enabled ? 'Disable' : 'Enable', () => setEnabled(skill, !skill.enabled));
  const priority = el('input');
  priority.type = 'number';
  priority.value = String(skill.priority);
  priority.setAttribute('aria-label', 'Skill priority');
  const savePriority = button('Set priority', () => setEnabled(skill, skill.enabled, Number(priority.value)));
  const remove = button('Delete', () => confirmDelete(skill), 'skills-button skills-danger');
  remove.disabled = !skill.editable;
  state.ui.controls.replaceChildren(toggle, priority, savePriority, remove);
  try {
    const data = await request(`/${encodeURIComponent(name)}/tree`);
    if (
      epoch !== state.selectionEpoch
      || state.selected !== name
      || currentAgent() !== agent
    ) return;
    renderTree(data.entries || []);
  } catch (error) {
    if (
      epoch === state.selectionEpoch
      && state.selected === name
      && currentAgent() === agent
    ) setStatus(detail(error), true);
  }
}

function renderTree(entries) {
  const ui = state.ui;
  if (!ui) return;
  ui.tree.replaceChildren();
  const root = { children: new Map() };
  for (const entry of entries) {
    const parts = entry.path.split('/');
    let parent = root;
    for (let index = 0; index < parts.length; index += 1) {
      const name = parts[index];
      if (!parent.children.has(name)) {
        parent.children.set(name, {
          name,
          path: parts.slice(0, index + 1).join('/'),
          type: index === parts.length - 1 ? entry.type : 'directory',
          executionRisk: false,
          children: new Map(),
        });
      }
      parent = parent.children.get(name);
      if (index === parts.length - 1) {
        parent.type = entry.type;
        parent.executionRisk = Boolean(entry.execution_risk);
      }
    }
  }

  const appendNodes = (container, nodes) => {
    const ordered = [...nodes.values()].sort((left, right) => left.name.localeCompare(right.name));
    for (const node of ordered) {
      const row = el('li');
      if (node.type === 'file') {
        const open = button(node.name, () => openFile(node.path));
        open.title = node.path;
        if (node.executionRisk) open.append(document.createTextNode(' ⚠'));
        row.append(open);
      } else {
        const children = el('ul', 'skills-tree skills-tree-nested');
        children.hidden = true;
        const toggle = button(`▸ ${node.name}`, () => {
          children.hidden = !children.hidden;
          toggle.textContent = `${children.hidden ? '▸' : '▾'} ${node.name}`;
          toggle.setAttribute('aria-expanded', String(!children.hidden));
        });
        toggle.setAttribute('aria-expanded', 'false');
        appendNodes(children, node.children);
        row.append(toggle, children);
      }
      container.append(row);
    }
  };
  appendNodes(ui.tree, root.children);
}

async function openFile(path) {
  if (!state.selected || !state.ui) return;
  const owner = { agent: currentAgent(), name: state.selected, path };
  const epoch = ++state.fileEpoch;
  state.path = null;
  state.editorOwner = null;
  state.ui.editor.value = '';
  state.ui.editor.disabled = true;
  state.ui.save.disabled = true;
  state.ui.warning.hidden = true;
  try {
    const file = await request(`/${encodeURIComponent(owner.name)}/file?path=${encodeURIComponent(path)}`);
    if (
      epoch !== state.fileEpoch
      || state.selected !== owner.name
      || currentAgent() !== owner.agent
    ) return;
    state.path = path;
    state.editorOwner = owner;
    state.ui.editor.value = file.content;
    state.ui.editor.disabled = !file.editable;
    state.ui.save.disabled = !file.editable;
    state.ui.warning.hidden = !file.execution_risk;
    setStatus(`Opened ${owner.name}/${path}.`);
  } catch (error) {
    if (
      epoch === state.fileEpoch
      && state.selected === owner.name
      && currentAgent() === owner.agent
    ) setStatus(detail(error), true);
  }
}

async function saveFile() {
  if (!state.ui || !state.editorOwner) return;
  const owner = state.editorOwner;
  if (
    state.selected !== owner.name
    || state.path !== owner.path
    || currentAgent() !== owner.agent
  ) {
    clearSelection();
    setStatus('Selection changed; reopen the file before saving.', true);
    return;
  }
  const content = state.ui.editor.value;
  state.ui.save.disabled = true;
  try {
    await request(`/${encodeURIComponent(owner.name)}/file`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: owner.path, content }),
    });
    if (
      state.editorOwner !== owner
      || state.selected !== owner.name
      || currentAgent() !== owner.agent
    ) return;
    await loadCatalog();
    if (state.editorOwner === owner && currentAgent() === owner.agent) {
      state.ui.save.disabled = false;
      setStatus(`Saved ${owner.name}/${owner.path}. No code was executed.`);
    }
  } catch (error) {
    if (state.editorOwner === owner && currentAgent() === owner.agent) {
      state.ui.save.disabled = false;
      setStatus(`Save rejected: ${detail(error)}`, true);
    }
  }
}

async function setEnabled(skill, enabled, priority = null) {
  const agent = currentAgent();
  try {
    await request(`/${encodeURIComponent(skill.name)}/state`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled, priority }),
    });
    if (currentAgent() !== agent) return;
    await loadCatalog();
    if (currentAgent() !== agent) return;
    await selectSkill(skill.name);
    setStatus(priority === null
      ? `${enabled ? 'Enabled' : 'Disabled'} ${skill.name}.`
      : `Updated ${skill.name} priority to ${priority}.`);
  } catch (error) {
    if (currentAgent() === agent) setStatus(detail(error), true);
  }
}

function confirmDelete(skill) {
  if (!skill.editable || !state.ui) return;
  state.ui.deleteDialog.dataset.skill = skill.name;
  state.ui.deleteDialog.showModal();
}

const panelDefinition = {
  panelId: PANEL_ID,
  label: 'Skills',
  icon: 'ki ki-book-open',
  before: 'features',
  gate: () => state.available,
  render: mount,
};

registerPanel(panelDefinition);

async function reconcileAvailability() {
  const agent = currentAgent();
  const epoch = ++state.availabilityEpoch;
  state.available = false;
  syncNav();
  try {
    await request('');
  } catch (_) {
    return;
  }
  if (epoch !== state.availabilityEpoch || currentAgent() !== agent) return;
  state.available = true;
  syncNav();
}

bus.on('panel:shown', (payload) => {
  if (payload?.panelId === PANEL_ID) {
    state.active = true;
    loadCatalog();
  }
});

bus.on('panel:hidden', (payload) => {
  if (payload?.panelId === PANEL_ID) state.active = false;
});

bus.on('agent:switch', () => {
  state.availabilityEpoch += 1;
  state.available = false;
  state.catalogEpoch += 1;
  state.catalog = null;
  if (state.ui?.createDialog.open) state.ui.createDialog.close('agent-switch');
  if (state.ui?.deleteDialog.open) state.ui.deleteDialog.close('agent-switch');
  if (state.ui?.deleteDialog) delete state.ui.deleteDialog.dataset.skill;
  clearSelection();
  renderCatalog();
  syncNav();
  void reconcileAvailability();
});

if (typeof globalThis.addEventListener === 'function') {
  globalThis.addEventListener('capabilities:changed', () => {
    void reconcileAvailability();
  });
}

export { loadCatalog, reconcileAvailability, reloadCatalog };
