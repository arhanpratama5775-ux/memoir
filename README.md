# memoir

your coding autobiography, written by data.

you know that feeling when you look at a repo and think "what the hell happened here?" memoir answers that. it reads your git history, finds the patterns nobody notices, and writes the actual story of your codebase.

the 3am commits. the file that's been "fixed" 14 times. the month where everyone just stopped committing. that's not random noise. that's a story. and nobody reads it because `git log` is a garbage way to tell stories.

memoir reads it for you.

---

## what it does

it goes through your entire commit history, finds stuff you probably didn't realize was happening, and writes it out like a book about your repo. not a metrics report. a story.

- **8 pattern detectors** that find recurring fixes, tech debt piling up, burnout signals, learning curves, code ownership black holes, anti-patterns, weird work hours, and code that keeps getting rewritten for no reason
- **5 crisis forecasters** that try to predict burnout, tech debt blowups, bus factor risk, code getting unmaintainable, and projects slowly dying
- **narrative chapters** that read like a book, not a spreadsheet. prologue, pattern breakdowns, milestones, crisis reports, current state, forecasts
- **health score** from 0 to 100. one number. tells you if your repo is healthy or on fire
- **3 export formats** so you can read it however you want

## install

```
pip install memoir
```

or build from source:

```
git clone https://github.com/arhanpratama5775-ux/memoir.git
cd memoir
pip install -e .
```

## the 30 second version

```
memoir scan
```

that's it. point it at a repo (defaults to current dir) and it does everything. analyzes history, finds patterns, predicts risks, writes the story, exports to `.memoir/exports/`.

## commands

### scan

the big one. runs the full pipeline.

```
memoir scan                              # current directory
memoir scan --repo /path/to/project      # specific repo
memoir scan --author "jane"              # filter by author
memoir scan --since 2024-01-01           # date range
memoir scan -f markdown -f html -f json  # multiple formats
memoir scan --refresh                    # ignore cache, redo everything
memoir scan --ai                         # AI-enhanced writing (needs API key)
```

### patterns

shows what patterns got detected. the stuff hiding in your code that you don't notice day to day.

```
memoir patterns
```

you'll see things like "payment/handler.py has been fixed 7 times in 4 months" or "after-hours commits went from 12% to 34%" or "one person owns 95% of the auth module." stuff that should probably worry you.

### forecast

risk predictions with actual numbers. not vibes.

```
memoir forecast
```

"67% chance of burnout in 6-8 weeks if the current trend continues." that kind of thing. based on where your indicators are heading, not gut feelings.

### health

quick pulse check. one number plus key stats.

```
memoir health
```

### export

generate files from cached data. run scan first.

```
memoir export -f markdown -f html
```

### status

what's cached, when it was last updated, how much disk it's using.

```
memoir status
```

### config

```
memoir config                              # see everything
memoir config ai_api_key sk-xxxx           # set API key
memoir config --global ai_api_key sk-xxxx  # set globally
memoir config after_hours_start 22         # change what counts as "after hours"
```

### reset

blows away all cached data. asks before doing it.

```
memoir reset
```

## the 8 detectors

| detector | what it catches |
|---|---|
| recurring fix | files that keep getting patched over and over |
| technical debt | growing TODO/FIXME counts, churn with no improvement |
| burnout indicator | after-hours creep, vague messages increasing, burst-silence cycles |
| learning curve | mistakes going down over time, practices improving |
| code ownership | bus factor risk, one person knowing too much |
| anti-pattern | god objects, tight coupling, complexity accelerating |
| irregular hours | late nights becoming normal, weekends becoming normal |
| high churn | code that keeps getting rewritten without moving forward |

## the 5 forecasts

| forecast | what it predicts |
|---|---|
| burnout | is the work pattern heading toward a crash |
| tech debt crisis | when does the debt become unmanageable |
| bus factor | what happens if key people disappear |
| maintainability | when does the codebase become too hard to work with |
| stagnation | is the project slowly dying |

## AI enhancement (optional)

by default, memoir uses templates. they work fine. the facts are all there, the data is accurate.

but if you want the writing to read more naturally, like an actual book instead of a filled-in form, you can plug in an AI model. it only touches the writing style. the data stays the same. if the API call fails, it falls back to templates silently. your memoir never depends on an external service.

```
memoir config ai_api_key sk-your-key
memoir config ai_model gpt-4o-mini          # default
memoir config ai_base_url https://...        # if you use a different provider
```

or just set the environment variable:

```
export MEMOIR_AI_API_KEY=sk-your-key
```

then scan with `--ai`:

```
memoir scan --ai
```

works with any OpenAI-compatible API. set `ai_base_url` for non-OpenAI providers.

## config reference

| key | default | what it does |
|---|---|---|
| `ai_api_key` | not set | API key for enhanced narratives |
| `ai_model` | gpt-4o-mini | which model to call |
| `ai_base_url` | not set | custom endpoint |
| `after_hours_start` | 20 | when "after hours" begins (8pm) |
| `after_hours_end` | 7 | when "after hours" ends (7am) |
| `min_pattern_occurrences` | 3 | minimum data points before calling something a pattern |
| `forecast_confidence_threshold` | 0.3 | minimum confidence to issue a forecast |
| `author_filter` | not set | only look at commits from this author |
| `export_formats` | ["markdown"] | default formats |

## extending it

want to add your own detector or exporter? the codebase is built for it.

- new pattern detector: add a `detect_xxx()` method in `memoir/core/pattern_detector.py`, return a `Pattern` object with real evidence, and it gets picked up automatically
- new forecast model: same deal in `memoir/core/crisis_forecast.py`, return a `Forecast` object
- new export format: add a class in `memoir/exporters/`
- new chapter template: add it in `memoir/core/narrator.py`

the interface is the same across all detectors and forecasters. return the right object type with actual data and the pipeline handles the rest.

## privacy

everything stays local. all data is stored in `.memoir/` inside your repo. no cloud, no server, no telemetry, no tracking. your git history never leaves your machine.

## why this exists

every line of code you write is autobiography. but git logs are formatted like receipts, not stories. nobody sits down and reads 2000 commit messages to understand what happened to a project.

the patterns in your code aren't random. they're you. your habits, your stress, your learning, your weekends that turned into workdays. all sitting there in timestamps and file change counts. memoir just makes it legible.

the honest thing about a developer isn't what they write in commit messages. it's when they commit, what they rewrite, and what they keep fixing. data doesn't care about looking smart. it just says: this is what happened. now you can read it.

## license

MIT. do whatever you want with it.
