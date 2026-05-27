# memoir

your coding autobiography, written by data.

you know that feeling when you join a project and someone tells you "yeah don't touch that file, nobody knows why it works but it does"? memoir reads your git history and writes the story behind those files. the real story. not what the README claims. what the commits actually show.

every repo has a personality. the 3am commits, the files that keep getting "fixed" every other week, the quiet periods where nobody touched anything. that's not random. that's a story. and nobody's reading it because git log is unreadable and nobody has time to dig through 2000 commits.

memoir digs for you.

## what it does

it reads your entire git history, finds patterns you didn't know existed, and writes a narrative about your codebase. like a biography, but for code.

- **pattern detection** - finds recurring fixes, technical debt, burnout signals, learning curves, code ownership issues, anti-patterns, irregular work hours, and unstable code areas. not vibes. actual statistical patterns from your commit data.
- **crisis forecasting** - predicts burnout, tech debt crisis, bus factor risk, maintainability decline, and project stagnation before they happen. based on trend extrapolation, not guessing.
- **narrative generation** - turns data into readable chapters. like a book about your repo. with a prologue, pattern chapters, milestones, crisis reports, current state, and forecasts.
- **health score** - one number, 0-100, that tells you how healthy your codebase actually is. based on patterns, forecasts, work patterns, and code quality metrics.
- **export to anything** - markdown, json, html. the html export looks like a proper documentation site with dark mode and everything.

## install

```bash
pip install memoir
```

or from source:

```bash
git clone https://github.com/arhanpratama5775-ux/memoir.git
cd memoir
pip install -e .
```

## quick start

```bash
# analyze your repo and generate a memoir
memoir scan

# that's it. seriously.
```

this will analyze the git repo in your current directory, detect patterns, forecast risks, generate narrative chapters, and export everything to `.memoir/exports/`.

## commands

### scan - the main thing

```bash
# basic scan of current directory
memoir scan

# analyze a specific repo
memoir scan --repo /path/to/repo

# filter by author
memoir scan --author "your name"

# date range
memoir scan --since 2024-01-01 --until 2024-12-31

# export to multiple formats
memoir scan -f markdown -f html -f json

# force refresh (ignore cache)
memoir scan --refresh

# enable AI-enhanced narratives (needs API key, see config below)
memoir scan --ai
```

### patterns - what's hiding in your code

```bash
memoir patterns
```

shows all detected patterns in a table. you'll see things like:

- "payment/handler.py has been fixed 7 times across 4 months" (recurring fix)
- "after-hours commits increased from 12% to 34% over the last quarter" (burnout signal)
- "one author owns 95% of the auth module" (bus factor risk)

### forecast - what's coming

```bash
memoir forecast
```

shows risk forecasts with probabilities and timelines. like weather predictions but for your codebase. "67% chance of burnout in 6-8 weeks if current trend continues" kind of thing.

### health - quick check

```bash
memoir health
```

compact dashboard. health score, top risks, key stats. good for a quick pulse check.

### export - generate files

```bash
memoir export -f markdown -f html
```

exports your cached memoir. run `scan` first.

### status - what's cached

```bash
memoir status
```

shows what data is stored, when it was last updated, and how much space it takes.

### config - settings

```bash
# see all config
memoir config

# set your AI API key for enhanced narratives
memoir config ai_api_key sk-xxxx

# set it globally (applies to all repos)
memoir config --global ai_api_key sk-xxxx

# change after-hours definition (default: 8pm-7am)
memoir config after_hours_start 22
memoir config after_hours_end 8
```

### reset - clean slate

```bash
memoir reset
```

deletes all cached data. asks for confirmation first.

## how it works

under the hood:

1. **git analysis** - reads every commit, every file change, every message. extracts timing, patterns, code churn, author info, work schedules.
2. **pattern detection** - 8 different detectors look for specific patterns. each one requires hard evidence (minimum 3-5 occurrences) and computes confidence scores. no vibes, no guessing.
3. **crisis forecasting** - takes detected patterns and git trends, extrapolates them forward. computes probabilities based on how close indicators are to critical thresholds and how fast they're moving.
4. **narrative generation** - templates turn data into readable chapters. every claim in the narrative is backed by a real data point. optionally, AI can enhance the prose (but the facts stay the same).
5. **health scoring** - weighted combination of pattern severity, forecast risk levels, work pattern health, and code quality indicators. 100 = pristine, 0 = abandon hope.

all data is stored locally in `.memoir/` inside your repo. no cloud, no server, no tracking. your git history stays on your machine.

## the 8 pattern detectors

| pattern | what it finds |
|---------|--------------|
| recurring fix | files/areas that keep getting fixed over and over |
| technical debt | growing TODO/FIXME counts, churn without improvement |
| burnout indicator | after-hours creep, message quality decline, burst-silence cycles |
| learning curve | decreasing mistake frequency, improving practices |
| code ownership | bus factor risk, single-author dominance |
| anti-pattern | god objects, file coupling, accelerating complexity |
| irregular hours | late nights increasing, weekends increasing, no rest days |
| high churn | unstable code areas, constant rewrites without progress |

## the 5 forecasts

| forecast | what it predicts |
|----------|----------------|
| burnout | when your work pattern is heading toward burnout |
| tech debt crisis | when debt will become unmanageable |
| bus factor | what happens if key people leave |
| maintainability | when the codebase becomes too hard to maintain |
| stagnation | when the project is slowly dying |

## optional AI enhancement

by default, memoir uses templates to generate narratives. they're readable and data-accurate. but if you want prose that reads more like a novel and less like a report, you can enable AI enhancement.

```bash
# set your API key
memoir config --global ai_api_key sk-your-key-here

# or use environment variable
export MEMOIR_AI_API_KEY=sk-your-key-here

# scan with AI
memoir scan --ai
```

supports any OpenAI-compatible API. set `ai_base_url` if you're using a different provider.

the AI only enhances the writing style. all facts and data points come from the analysis. if the AI call fails, it silently falls back to templates. your memoir never depends on an API being up.

## config options

| key | default | what it does |
|-----|---------|-------------|
| `ai_api_key` | none | API key for AI-enhanced narratives |
| `ai_model` | gpt-4o-mini | model to use |
| `ai_base_url` | none | custom API endpoint |
| `after_hours_start` | 20 | hour when "after hours" begins (8pm) |
| `after_hours_end` | 7 | hour when "after hours" ends (7am) |
| `min_pattern_occurrences` | 3 | minimum data points to count as a pattern |
| `forecast_confidence_threshold` | 0.3 | minimum confidence to issue a forecast |
| `author_filter` | none | only analyze commits by this author |
| `export_formats` | ["markdown"] | default export formats |

## adding your own stuff

want to add a custom pattern detector? or a new export format? the codebase is structured for extension:

- `memoir/core/pattern_detector.py` - add a new `detect_xxx()` method
- `memoir/core/crisis_forecast.py` - add a new `forecast_xxx()` method
- `memoir/exporters/` - add a new exporter class
- `memoir/core/narrator.py` - add new chapter templates

each pattern detector and forecaster follows the same interface. return a `Pattern` or `Forecast` object with real evidence and the rest of the pipeline picks it up automatically.

## philosophy

every line of code you write is autobiography. but nobody reads git logs because they're formatted as receipts, not stories. memoir converts receipts into stories.

the patterns in your code aren't random. they're you. your habits, your fears, your learning, your exhaustion. all hiding in commit timestamps and file change frequencies. memoir makes the invisible visible.

the most honest thing about you isn't what you say in commit messages. it's when you commit, what you rewrite, and what you keep fixing. data doesn't have an ego. it doesn't care if you look smart or not. it just says: this is what you did. now read it.

## license

MIT
