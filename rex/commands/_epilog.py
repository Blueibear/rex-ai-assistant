"""Help epilog for the Rex CLI top-level parser (US-REM-027)."""

CLI_EPILOG = """
Examples:
  rex doctor
  rex doctor -v
  rex chat
  rex version
  rex tools
  rex tools -v
  rex run-workflow workflow.json
  rex run-workflow workflow.json --dry-run
  rex run-workflow workflow.json --resume
  rex approvals
  rex approvals --approve <id>
  rex workflows

Planning and execution commands:
  rex plan "send monthly newsletter"
  rex plan "check weather in Dallas" --execute
  rex plan "turn on living room lights" --save
  rex executor resume <workflow_id>

Memory commands:
  rex memory recent 5
  rex memory add facts '{"key":"value"}'
  rex memory add secrets '{"api":"key"}' --sensitive --ttl=7d
  rex memory search keyword
  rex memory search --category preferences
  rex memory forget <entry_id>
  rex memory stats

Knowledge base commands:
  rex kb ingest /path/to/file.txt --title "My Doc" --tags notes,project
  rex kb search "query"
  rex kb search "query" -v
  rex kb list
  rex kb show <doc_id>
  rex kb delete <doc_id>
  rex kb cite "phrase"
  rex kb tags

Scheduler commands:
  rex scheduler init
  rex scheduler list
  rex scheduler run <job_id>

Email commands:
  rex email unread
  rex email unread --limit 5
  rex email unread -v

Calendar commands:
  rex calendar upcoming
  rex calendar upcoming --days 14
  rex calendar upcoming --conflicts

Computer commands:
  rex pc list
  rex pc list --all
  rex pc status --id desktop
  rex pc run --id desktop --yes -- whoami
  rex pc run --id desktop --yes -- ipconfig

WordPress commands:
  rex wp health --site myblog

WooCommerce commands:
  rex wc orders list --site myshop
  rex wc orders list --site myshop --status pending --limit 20
  rex wc orders set-status --site myshop --order-id 101 --status completed
  rex wc orders set-status --site myshop --order-id 101 --status completed --yes
  rex wc products list --site myshop
  rex wc products list --site myshop --low-stock
  rex wc coupons create --site myshop --code SAVE10 --amount 10 --type percent
  rex wc coupons create --site myshop --code SAVE10 --amount 10 --type percent --yes
  rex wc coupons disable --site myshop --coupon-id 55
  rex wc coupons disable --site myshop --coupon-id 55 --yes

Home Assistant commands:
  rex ha tts test
  rex ha tts test --message "Hello from Rex" --entity-id media_player.living_room

For more information, visit: https://github.com/Blueibear/rex-ai-assistant
"""
