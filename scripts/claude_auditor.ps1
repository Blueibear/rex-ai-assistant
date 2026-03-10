param([string]$PacketFile)

$packet = Get-Content -Raw $PacketFile

$prompt = @"
You are the AUDITOR agent.

Verify that the task was correctly completed.

Check:

- code correctness
- tests
- task board updates

If the implementation is wrong:
mark the task as TODO again.

Task packet:
$packet
"@

$prompt | claude