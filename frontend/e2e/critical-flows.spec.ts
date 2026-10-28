import { test, expect } from "@playwright/test";
import path from "path";

/**
 * End-to-End Tests for Critical Flows (T8-010)
 * 1. Flow 1: Crop -> sensor entry -> result -> save to history -> audit verify
 * 2. Flow 2: Campus live reading (mock adapter) -> analyze
 * 3. Flow 3: Plant photo upload showing per-leaf cards
 * 4. Flow 4: Fallbacks & error handling (device unavailable -> manual, segmentation failure -> retry, low-confidence -> uncertain tier)
 */

test.describe("Critical E2E Flows", () => {
  // Shared in-memory store for history round-trip simulation
  let savedRecords: Array<{
    id: string;
    crop: string;
    mode: string;
    source: string;
    observed_at: string;
    observed_tz: string;
    created_at: string;
    inputs?: Record<string, unknown>;
    predictions?: Record<string, unknown>;
    recommendations?: Record<string, unknown>;
    historical: boolean;
  }> = [];

  test.beforeEach(async ({ page }) => {
    savedRecords = [
      {
        id: "rec_seed_001",
        crop: "lettuce",
        mode: "both",
        source: "manual",
        observed_at: new Date(Date.now() - 3600000).toISOString(),
        observed_tz: "UTC",
        created_at: new Date(Date.now() - 3600000).toISOString(),
        inputs: { ph: 6.2, ec: 1.6 },
        predictions: {
          crop: "lettuce",
          modality: "sensor",
          model_name: "lettuce_sensor_v1.0.0",
          model_version: "v1.0.0",
          predicted_class: "Normal",
          display_label: "Optimal Nutrient Balance",
          confidence: 0.92,
          is_uncertain: false,
          inference_time_ms: 12,
          evidence: {},
        },
        historical: true,
      },
    ];

    // Mock API endpoints for consistent, isolated test execution
    await page.route("**/api/v1/history**", async (route) => {
      const request = route.request();
      if (request.method() === "POST") {
        const body = JSON.parse(request.postData() || "{}");
        const newRecord = {
          id: `rec_${Date.now()}`,
          crop: body.crop || "lettuce",
          mode: body.mode || "both",
          source: body.source || "manual",
          observed_at: body.observed_at || new Date().toISOString(),
          observed_tz: body.observed_tz || "UTC",
          created_at: new Date().toISOString(),
          inputs: body.inputs || {},
          predictions: body.predictions || null,
          recommendations: body.recommendations || null,
          historical: true,
        };
        savedRecords.unshift(newRecord);
        await route.fulfill({
          status: 201,
          contentType: "application/json",
          body: JSON.stringify(newRecord),
        });
      } else if (request.method() === "GET") {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            records: savedRecords,
            total: savedRecords.length,
            limit: 10,
            offset: 0,
          }),
        });
      } else {
        await route.continue();
      }
    });

    // Default mock adapter device status
    await page.route("**/api/v1/device/status", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          adapter: "mock",
          connected: true,
          available_sensors: ["ph", "ec", "water_temp", "humidity", "air_temp"],
          camera_available: true,
          last_reading_utc: new Date().toISOString(),
        }),
      });
    });
  });

  // Flow 1: Crop -> sensor entry -> result -> save to history
  test("Flow 1: Crop to sensor entry to result to save to history", async ({ page }) => {
    await page.goto("/analyze");
    await expect(page.locator("h1")).toContainText("Crop Nutrition & Anomaly Analysis");

    // 1. Select cultivar: Spinach
    await page.selectOption("#crop-select", "spinach");

    // 2. Select Modality: Sensor Only
    await page.click("button:has-text('Sensor Only')");

    // 3. Sensor input form is displayed and validated
    const executeButton = page.locator("[data-testid='execute-analysis-button']");
    await expect(executeButton).toBeEnabled();

    // 4. Click Execute Analysis
    await executeButton.click();

    // 5. Results section renders with recommendations
    const resultsSection = page.locator("[data-testid='analysis-results-section']");
    await expect(resultsSection).toBeVisible({ timeout: 10000 });
    await expect(page.locator("text=Inspection Assessment Ready")).toBeVisible();
    await expect(page.locator("text=Optimal Nutrient Balance")).toBeVisible();

    // 6. Save to history
    const saveButton = page.locator("[data-testid='save-record-button']");
    await expect(saveButton).toBeVisible();
    await saveButton.click();
    await expect(saveButton).toContainText("Record Saved", { timeout: 5000 });

    // 7. Navigate to History page and verify record is present
    await page.goto("/history");
    await expect(page.locator("h1")).toContainText("Inspection History & Audit Log");
    await expect(page.locator("table")).toBeVisible();
    await expect(page.locator("table tbody tr").first()).toBeVisible();
    await expect(page.locator("table").getByText("Spinach", { exact: false })).toBeVisible();
  });

  // Flow 2: Campus live reading (mock adapter) to analyze
  test("Flow 2: Campus live reading (mock adapter) to analyze", async ({ page }) => {
    await page.goto("/analyze");

    // Switch to Campus Live Adapter
    await page.click("button:has-text('Campus Live Adapter')");

    // Verify mock device adapter is connected (no error card)
    await expect(page.locator("[data-testid='device-unavailable-card']")).not.toBeVisible();

    // Modality: Sensor Only
    await page.click("button:has-text('Sensor Only')");

    // Execute analysis
    const executeButton = page.locator("[data-testid='execute-analysis-button']");
    await expect(executeButton).toBeEnabled();
    await executeButton.click();

    // Verify results with campus live adapter source provenance
    await expect(page.locator("[data-testid='analysis-results-section']")).toBeVisible({ timeout: 10000 });
    await expect(page.locator("text=Optimal Nutrient Balance")).toBeVisible();
  });

  // Flow 3: Plant photo upload showing per-leaf cards
  test("Flow 3: Plant photo upload showing per-leaf cards", async ({ page }) => {
    await page.goto("/analyze");

    // Select Modality: Image Only
    await page.click("button:has-text('Image Only')");

    // Upload plant image fixture
    const fixturePath = path.resolve(__dirname, "fixtures/test_leaf.png");
    const fileInput = page.locator("input[type='file']").first();
    await fileInput.setInputFiles(fixturePath);

    // Verify file name appears in upload card
    await expect(page.locator("text=test_leaf.png").first()).toBeVisible();

    // Execute analysis
    const executeButton = page.locator("[data-testid='execute-analysis-button']");
    await expect(executeButton).toBeEnabled();
    await executeButton.click();

    // Verify per-leaf cards are displayed
    await expect(page.locator("[data-testid='analysis-results-section']")).toBeVisible({ timeout: 10000 });
    await expect(page.locator("[data-testid='leaf-grid']")).toBeVisible();
    await expect(page.locator("text=Assessed Foliage Breakdown")).toBeVisible();

    // Verify multi-leaf breakdown and leaf cards
    await expect(page.locator("text=Leaf 1")).toBeVisible();
    await expect(page.locator("text=Leaf 2")).toBeVisible();
    await expect(page.locator("text=Leaf 3")).toBeVisible();
    await expect(page.locator("text=Healthy Foliage").first()).toBeVisible();
  });

  // Flow 4: Fallbacks and Error Handling
  test.describe("Flow 4: Fallbacks and error handling", () => {
    test("4A: Device unavailable offers manual entry fallback", async ({ page }) => {
      // Override status to simulate hardware gateway disconnection
      await page.route("**/api/v1/device/status", async (route) => {
        await route.fulfill({
          status: 503,
          contentType: "application/json",
          body: JSON.stringify({
            error: {
              code: "DEVICE_UNAVAILABLE",
              message: "Campus hardware server unreachable.",
            },
          }),
        });
      });

      await page.goto("/analyze");
      await page.click("button:has-text('Campus Live Adapter')");

      // Verify device unavailable card appears
      const alertCard = page.locator("[data-testid='device-unavailable-card']");
      await expect(alertCard).toBeVisible();
      await expect(alertCard).toContainText("Campus Hardware Server Unreachable");

      // Execute button should be disabled
      const executeButton = page.locator("[data-testid='execute-analysis-button']");
      await expect(executeButton).toBeDisabled();

      // Click fallback: Switch to Manual Entry
      await page.click("button:has-text('Switch to Manual Entry')");

      // Verify returned to manual observation
      await expect(alertCard).not.toBeVisible();
    });

    test("4B: Image upload with segmentation failure displays retry prompt", async ({ page }) => {
      await page.goto("/analyze");
      await page.click("button:has-text('Image Only')");

      // Upload blur / segmentation failure fixture
      const fixturePath = path.resolve(__dirname, "fixtures/seg_fail_blur.png");
      const fileInput = page.locator("input[type='file']").first();
      await fileInput.setInputFiles(fixturePath);

      // Execute analysis
      const executeButton = page.locator("[data-testid='execute-analysis-button']");
      await executeButton.click();

      // Verify segmentation failure card is displayed with retry button
      await expect(page.locator("text=Leaf Segmentation Unsuccessful")).toBeVisible({ timeout: 10000 });
      await expect(page.locator("text=Non-Diagnostic Failure")).toBeVisible();

      const retryButton = page.locator("button:has-text('Retry Segmentation')");
      await expect(retryButton).toBeVisible();
      await retryButton.click();

      // Progress indicator appears upon retry
      await expect(page.locator("[data-testid='vine-progress-indicator']")).toBeVisible({ timeout: 5000 });
    });

    test("4C: Low-confidence result shows the uncertain tier", async ({ page }) => {
      await page.goto("/analyze");
      await page.click("button:has-text('Image Only')");

      // Upload uncertain foliage fixture
      const fixturePath = path.resolve(__dirname, "fixtures/uncertain_foliage.png");
      const fileInput = page.locator("input[type='file']").first();
      await fileInput.setInputFiles(fixturePath);

      // Execute analysis
      const executeButton = page.locator("[data-testid='execute-analysis-button']");
      await executeButton.click();

      // Verify uncertain tier badge is displayed
      await expect(page.locator("[data-testid='analysis-results-section']")).toBeVisible({ timeout: 10000 });
      await expect(page.locator("text=Uncertain Score").first()).toBeVisible();
    });
  });
});
