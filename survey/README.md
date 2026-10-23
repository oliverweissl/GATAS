# Realism survey

1. `python survey/build_survey.py` (project root) writes the public site to `docs/` (anonymous, loudness-normalized audios, `items.json`)
   and the secret key to `survey/private/key.json` (git-ignored: keep it out of the repository).
2. Collect the answers: create a Google Sheet, Extensions > Apps Script, paste, deploy as web app (execute as me, access: anyone), and put
   the `/exec` URL into `docs/config.js`:
   ```js
   function doPost(e) {
     SpreadsheetApp.getActiveSheet().appendRow([new Date(), e.postData.contents]);  // one JSON session per row
     return ContentService.createTextOutput("ok");
   }
   ```
3. Publish: commit `docs/`, then GitHub > Settings > Pages > Deploy from branch > `main` / `/docs`.
   Participants with a Prolific id: `https://<user>.github.io/<repo>/?pid=<id>`.
4. Local test: `python survey/server.py 8765` (answers go to `survey/responses.jsonl`).
