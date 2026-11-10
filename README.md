# Internship Radar

A bot that finds new internship postings the day they go live, scores them against my resumes, and sends the good ones to a private Discord channel. It never applies for me. I read every match and apply by hand.

## Why

I was checking career pages every day, and job aggregators lag the employer's own page. I wanted to see a posting while it was still new, so I stopped refreshing tabs and built something that does it for me.

## Who built what

I wrote the spec, the product decisions and the milestone gates: human in the loop, a hard monthly spending cap, two resume variants, and a score threshold for drafts. I directed Claude Code through those gates, and it wrote the code. The decisions and the testing are mine. The Python is not.

## How it works

A GitHub Actions run every ~30 minutes:

1. **Fetches** the [SimplifyJobs Summer 2027 feed](https://github.com/SimplifyJobs/Summer2027-Internships) and 186 company boards (Greenhouse, Lever, Ashby) from `config/watchlist.yaml`.
2. **Normalizes** everything into one `Posting` model and merges duplicates. If Simplify and the employer both list a job, the employer's copy wins because it has the real publish time.
3. **Filters** for free before any model is called: intern-level titles, 2027 seasons, my locations (NYC area, NJ and CT in person, US remote, other states only with relocation), and no non-US, quant or hardware roles.
4. **Dedupes** against IDs already handled, stored on the `state` branch.
5. **Scores** each new match with Claude Haiku: which resume fits, a 0-100 score, reasons, gaps and eligibility flags, returned as strict JSON and validated with pydantic. Recent 👍/👎 reactions go back to the scorer as calibration examples.
6. **Posts** 60+ to Discord. 80+ also pings me, silently between 01:00 and 07:00 ET.
7. **Drafts on request.** React ✍️ on an 80+ card and the next run has Claude Sonnet write resume edits, a cover note and a referral message. Any suggested edit that doesn't quote a real line of my resume is dropped.

Problems go to a separate errors channel, at most once per 6 hours each. A daily heartbeat reports spend and detection lag.

## Guardrails

- **$15/month hard cap.** Every Claude call is checked against it, assuming the call's worst case, before it's made. A warning posts at $10 and there is also a daily call cap. At the cap, matches still post, unscored and silent, until the month resets.
- **Scoring is cheap, drafting is deliberate.** Scoring is about $0.001 per posting and a draft about $0.04 (estimates), and drafts only happen when I ask.
- **Job descriptions are untrusted input.** The model has no tools and must return schema-valid JSON. Description fetching follows only public `https` URLs and re-checks every redirect.
- **Public logs hold no private data.** Actions logs on a public repo are public, so runs log counts and posting IDs only. Never resume text, descriptions or model output.

## What I learned

- GitHub's scheduler was unreliable for this repo, so an external cron triggers runs. A dead-man's-switch alarm tells me when the pings stop.
- Check the budget before the call, not after. A ledger that tells you what you already spent is a receipt, not a cap.
- Free filters before paid ones. Most postings die before Claude ever sees them.

## Setup

**Secrets** (`gh secret set NAME`; values are never committed or logged):

| Secret | Purpose |
|---|---|
| `DISCORD_WEBHOOK_MATCHES` | Matches channel |
| `DISCORD_WEBHOOK_ERRORS` | Alerts and the heartbeat |
| `DISCORD_PING_USER_ID` | Who to mention on 80+ scores |
| `ANTHROPIC_API_KEY` | Claude API key |
| `RESUME_SWE_TEXT`, `RESUME_PM_TEXT` | Plain-text resumes (`gh secret set RESUME_SWE_TEXT < swe.txt`) |
| `HEALTHCHECK_PING_URL` | healthchecks.io ping URL (optional, recommended) |
| `DISCORD_BOT_TOKEN` | Read-only bot for 👍/👎 feedback and ✍️ drafts (optional; no drafts without it) |

**Trigger (cron-job.org).** Create a fine-grained GitHub token for this repo only, with Actions: Read and write. Then add a job that runs every 30 minutes and POSTs `{"ref":"main"}` to `https://api.github.com/repos/<owner>/internship-radar/actions/workflows/radar.yml/dispatches` with the headers `Authorization: Bearer <token>`, `Accept: application/vnd.github+json` and `X-GitHub-Api-Version: 2022-11-28`. A test should return 204. Extra triggers are harmless: runs queue one at a time and state prevents re-posting.

**Alarm (healthchecks.io).** Make a check with a 30 minute period and 90 minute grace, point its Discord integration at the errors channel, and store the ping URL as `HEALTHCHECK_PING_URL`. Failed runs ping `/fail`, and silence for ~2 hours alerts.

**Reaction bot (optional).** In the Discord Developer Portal, create an application and turn on **Message Content Intent**. Invite it with scope `bot` and only **View Channel** and **Read Message History**, then run `gh secret set DISCORD_BOT_TOKEN`. If the bot can't read the channel, scoring continues without examples and the errors channel says why.

## Running it

| Task | How |
|---|---|
| Is it running? | `gh run list --workflow radar.yml --limit 10`, the healthchecks.io dashboard, or the daily heartbeat |
| Runs stopped | Check cron-job.org's history. 401 means the token expired, so make a new one |
| "Board ... has failed N runs in a row" | The company probably changed job systems. Run `uv run python scripts/probe_boards.py`, review the watchlist, commit |
| Watchlist is stale (monthly reminder) | `uv run python scripts/probe_boards.py --discover`, review, commit |
| Next season | Point `sources.simplify.url` at the new SimplifyJobs repo and update `filters.seasons` |
| Try it locally | `uv run radar --dry-run` fetches and filters with no Claude calls, posts or state writes |
| Test a scoring change | `gh workflow run eval.yml` scores a live sample under several variants. It costs well under $1 |
| Pause | Disable the cron-job.org job and run `gh workflow disable radar.yml` |

Settings live in `config/settings.yaml` and handled IDs, spend and lag stats live on the `state` branch, as IDs and numbers only.

## Development

```sh
uv sync
uv run ruff check . && uv run ruff format --check .
uv run pytest
```

CI runs the same checks on every push. Tests use recorded fixtures and mocks, so no secrets are needed.
