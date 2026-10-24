# Realism survey

Every participant rates 32 audios: 2 for each of 16 configurations (4 cells of SUT x task, each with GATAS / ALIF / SMACK / clean audio), of
distinct sentences per task. Within a configuration the audios with the fewest answers so far are chosen, so every audio gets at least 5
answers with 100 respondents (simulated: min 5, max 11; 150 respondents: min 8). The balancing needs the endpoint below to return the counts.

1. `python survey/build_survey.py` (project root) writes the site to `docs/` and the key to `survey/private/key.json` (git-ignored).
2. Google Sheet: create a sheet, Extensions > Apps Script, paste the script below, Deploy > New deployment > Web app (execute as: me,
   access: anyone), copy the `/exec` URL into `docs/config.js` (`window.SURVEY_ENDPOINT`).
3. GitHub Pages serves the repository (root redirects to `docs/`); participants with Prolific ids: `?pid=<id>`.
4. Local test: `python survey/server.py 8765` (answers go to `survey/responses.jsonl`, endpoint `submit`).

```js
function sheet_() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  return ss.getSheetByName("responses") || ss.insertSheet("responses");
}

function doPost(e) {  // one row per rated audio
  const s = JSON.parse(e.postData.contents), sh = sheet_();
  if (sh.getLastRow() === 0)
    sh.appendRow(["time", "participant", "headphones", "item", "order", "realism", "understood_match", "plausible_transcription", "plays", "seconds"]);
  const rows = s.trials.map(t => [new Date(), s.participant, s.profile.headphones, t.item, t.order,
                                  t.realism, t.understood_match, t.plausible_transcription, t.plays, (t.t_end - t.t_start) / 1000]);
  if (rows.length) sh.getRange(sh.getLastRow() + 1, 1, rows.length, rows[0].length).setValues(rows);
  return ContentService.createTextOutput("ok");
}

function doGet() {  // number of answers per audio, used to balance the assignment
  const sh = sheet_(), n = sh.getLastRow(), counts = {};
  if (n > 1) sh.getRange(2, 5, n - 1, 1).getValues().forEach(r => counts[r[0]] = (counts[r[0]] || 0) + 1);
  return ContentService.createTextOutput(JSON.stringify(counts)).setMimeType(ContentService.MimeType.JSON);
}
```

Analysis: join the `item` column with `survey/private/key.json` (item -> cell, sentence, condition).
After changing the script, deploy a new version (Deploy > Manage deployments > Edit > New version); the `/exec` URL stays the same.
