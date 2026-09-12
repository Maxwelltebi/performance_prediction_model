// Exercise the real page script with a small DOM/network harness; no npm packages.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'frontend/index.html'), 'utf8');
const source = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const schema = JSON.parse(fs.readFileSync(path.join(root, 'public/form-schema.json'), 'utf8'));

async function page() {
  const elements = new Map();
  const requests = [];
  const timers = new Map();
  let timerId = 0;
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, {
      innerHTML: '', textContent: '', disabled: false, value: '', listeners: {},
      addEventListener(name, listener) { this.listeners[name] = listener; },
      reportValidity: () => true, querySelectorAll: () => [],
    });
    return elements.get(id);
  };
  schema.fields.forEach((f) => { element(f.name).value = String(f.default); });
  const context = vm.createContext({
    document: { getElementById: element }, AbortController,
    setTimeout(fn, ms) { const id = ++timerId; timers.set(id, { fn, ms }); return id; },
    clearTimeout(id) { timers.delete(id); }, requestAnimationFrame: (fn) => fn(),
    fetch: async (url) => {
      requests.push(url);
      assert.equal(url, '/form-schema.json', 'page load must not wake an API function');
      return { ok: true, json: async () => schema };
    },
  });
  await vm.runInContext(source, context);
  return { context, element, requests, timers,
    submit: () => element('form').listeners.submit({ preventDefault() {} }) };
}

test('form and buttons initialize using static schema only', async () => {
  const p = await page();
  assert.deepEqual(p.requests, ['/form-schema.json']);
  for (const field of schema.fields) assert.ok(p.element('fields').innerHTML.includes(`id="${field.name}"`));
  assert.equal(typeof p.element('sample').listeners.click, 'function');
  assert.equal(typeof p.element('reset').listeners.click, 'function');
});

test('prediction shows a wait message, then renders the result and clears timers', async () => {
  const p = await page();
  let resolve;
  p.context.fetch = (url, options) => {
    assert.equal(url, '/api/predict');
    assert.equal(Object.keys(JSON.parse(options.body)).length, 13);
    assert.equal(options.headers.Authorization, undefined);
    return new Promise((done) => { resolve = done; });
  };
  const pending = p.submit();
  assert.equal(p.element('submit').disabled, true);
  [...p.timers.values()].find((t) => t.ms === 5000).fn();
  assert.match(p.element('form-msg').innerHTML, /Still waiting/);
  resolve({ ok: true, json: async () => ({ grade: 'b', label: 'Very good', confidence: 0.68, breakdown: [] }) });
  await pending;
  assert.match(p.element('result').innerHTML, /Very good/);
  assert.equal(p.element('form-msg').innerHTML, '');
  assert.equal(p.element('submit').disabled, false);
  assert.equal(p.timers.size, 0);
});

test('HTML gateway errors produce a readable message and allow another submission', async () => {
  const p = await page();
  p.context.fetch = async () => ({ ok: false, status: 502, json: async () => { throw new SyntaxError(); } });
  await p.submit();
  assert.match(p.element('form-msg').innerHTML, /Request failed \(502\)/);
  assert.equal(p.element('submit').disabled, false);
  assert.equal(p.timers.size, 0);
});

test('a stalled prediction times out and restores the form', async () => {
  const p = await page();
  p.context.fetch = (url, options) => new Promise((resolve, reject) => {
    options.signal.addEventListener('abort', () => {
      const error = new Error('Aborted'); error.name = 'AbortError'; reject(error);
    });
  });
  const pending = p.submit();
  [...p.timers.values()].find((t) => t.ms === 90000).fn();
  await pending;
  assert.match(p.element('form-msg').innerHTML, /taking too long/);
  assert.equal(p.element('submit').disabled, false);
  assert.equal(p.timers.size, 0);
});
