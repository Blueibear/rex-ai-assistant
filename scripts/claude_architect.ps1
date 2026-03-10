param([string]$PacketFile)

$packet = Get-Content -Raw $PacketFile

$prompt = @"
You are the ARCHITECT agent.

Read the task packet.

Your job:
1. Understand the task
2. Identify the exact repo files needed
3. Append an ALLOWED_FILES section listing those files.

Task packet:
$packet
"@

$prompt | claude