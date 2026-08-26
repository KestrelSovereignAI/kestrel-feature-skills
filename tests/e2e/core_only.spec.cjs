const { test, expect } = require('@playwright/test');

test('core-only console has no dead procedural-skills panel or loader error', async ({ page }) => {
  test.skip(process.env.KESTREL_EXPECT_SKILLS !== '0', 'core-only suite');
  const apiKey = process.env.KESTREL_API_KEY || '';
  await page.addInitScript((key) => {
    if (key) globalThis.sessionStorage.setItem('kestrel_api_key', key);
  }, apiKey);
  const errors = [];
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });
  await page.goto('/');
  await page.waitForLoadState('networkidle');
  await expect(page.locator('.nav-tab[data-panel="procedural-skills"]')).toHaveCount(0);
  expect(errors.filter((message) => message.includes('procedural-skills'))).toEqual([]);
});
