# Message specs

Timestamps are always ISO 8601 format e.g. 2026-02-14T01:37:10Z.
`ts` is the abbreviation for timestamp.

## System

{ role: system, content: ..., ts: ... }

## User

{ role: user, content: ..., ts: ... }

## Assistant

{ role: assistant, content: ..., ts: ... }

## Thinking

{ type: thinking, content: ..., ts: ... }

## Tool call

{ type: tool_call, name: get_weather, id: ..., input: { ... } }

## Tool result

{ type: tool_result, id: ..., output: { ... }}
