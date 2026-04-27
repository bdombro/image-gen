_:
    @just --list

setup:
    uv tool install mflux

alchemist:
    bun run ts-gen/alchemist-sprite-gen.ts

room:
    bun run ts-gen/room-gen.ts