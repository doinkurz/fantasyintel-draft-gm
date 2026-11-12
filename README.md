# FantasyIntel Draft GM

A virtual general manager for fantasy football drafts, written in Python.
It takes your league's settings, projects every player under your scoring,
tracks the draft, recommends picks, and answers questions in a chat panel.

## What is in this folder

| File | What it does |
|---|---|
| `app.py` | The web app: setup screen, draft room, chat panel |
| `data.py` | Downloads free NFL data (nflverse, via `nflreadpy`) |
| `projections.py` | Turns that data into projected fantasy points |
| `draft.py` | Tracks the draft and ranks the best pick |
| `gm.py` | Answers questions and reads league rules documents (Google Gemini) |
| `requirements.txt` | The Python packages the app needs |
| `.streamlit/config.toml` | Colors |
| `.streamlit/secrets.toml.example` | Template for your private API key |
| `assets/logo.png` | Your logo |
| `tests/` | Offline checks that use made-up data |

## Step 6: test it on your computer (Windows, Anaconda)

1. Unzip the folder somewhere easy, for example `C:\fantasyintel-draft-gm`.
2. Add your Gemini key:
   - Open the `.streamlit` folder.
   - Make a copy of `secrets.toml.example` and name the copy `secrets.toml`.
   - Open `secrets.toml` in Notepad, replace `paste-your-key-here` with your
     key (keep the quotes), and save.
3. Open **Anaconda Prompt** from the Start menu and run these one at a time:

   ```
   cd C:\fantasyintel-draft-gm
   pip install -r requirements.txt
   python tests\test_engine.py
   streamlit run app.py
   ```

   The third line should end with `ALL ENGINE CHECKS PASSED`. The fourth
   opens the app in your browser. The first time you click **Start draft**
   it downloads NFL data, which takes about a minute.

To stop the app, press Ctrl+C in the Anaconda Prompt window.

## Step 7: upload to GitHub

1. On github.com, click **+** (top right) then **New repository**.
2. Name it `fantasyintel-draft-gm`, choose **Public**, click **Create repository**.
3. Click **uploading an existing file**.
4. Drag in everything from your folder **except the `.streamlit` folder**,
   then click **Commit changes**. Leaving that folder out keeps
   `secrets.toml`, which holds your private key, off GitHub.
5. Add the colors file by hand: click **Add file > Create new file**, type
   `.streamlit/config.toml` as the name, paste in the contents of your
   `config.toml`, and click **Commit changes**.

## Step 8: deploy on Streamlit Community Cloud

1. Go to share.streamlit.io and click **Create app**.
2. Choose your `fantasyintel-draft-gm` repository, branch `main`, file `app.py`.
3. Open **Advanced settings**. In the **Secrets** box paste this line with
   your real key:

   ```
   GEMINI_API_KEY = "your-key"
   ```

4. Click **Deploy**. After a few minutes you get an address ending in
   `.streamlit.app`.

## Step 9: embed in Wix

1. Open fantasyintel.com in the Wix Editor and go to the page for the tool.
2. Click **Add Elements > Embed Code > Embed a Site**.
3. Paste your app address with `?embed=true` on the end, for example
   `https://fantasyintel-draft-gm.streamlit.app/?embed=true`.
4. Stretch the element to full width and about 1,400 pixels tall, then
   **Publish**.

## How the projections work

Each player's projection starts from past performance and is nudged by three
factors. Every nudge is capped at 4% and shown in the Adjustments column.

| Factor | Source | How it is used |
|---|---|---|
| Past performance | Player stats, last three seasons | Per-game average, weighted 55% / 30% / 15% toward the most recent season |
| Offensive line | Team sack rate and yards per carry | Stand-in for a line rating: helps or hurts passers, receivers and backs |
| Coaching trend | Head coach's run/pass split | Pass-heavy coaches lift passers and receivers, run-heavy lift backs |
| Contract status | OverTheCap contract data | Players in the last year of a multi-year deal get +3% |

Recommendations rank players by projected points above a replacement-level
player at the same position, then adjust for your open roster slots and for
how much the position will thin out before your next pick.

## Known limits of this version

- Snake drafts only. Auction is selectable but runs as a snake draft.
- Picks are entered by hand. There is no live sync with Yahoo, ESPN or
  PrimeTime Draft yet.
- Rookies and players who changed teams in the offseason are not handled:
  projections use each player's most recent team and past NFL stats.
- The projection is a transparent formula. It has not been backtested.
- Dynasty leagues get one-season projections, with no age or future value.
- No user accounts. Each browser session is its own draft, and refreshing
  the page starts over.
