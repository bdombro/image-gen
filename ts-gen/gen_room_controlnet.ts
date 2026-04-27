#!/usr/bin/env bun

/*
Generates a ControlNet control image showing:
- Room walls as thick lines at image edges (goes edge-to-edge)
- A walking path connecting entry (bottom) to exit (top)

The image is saved at 512×512 to match the generation resolution,
with thick lines so the VAE preserves the control signal.
*/

import { createCanvas } from "canvas"

const SIZE = 512
const WALL = 16      // wall thickness
const PATH = 8       // path thickness
const GAP = 12       // gap in wall for doors

// Route: entry at bottom-center, L-shape to top-right
const route: [number, number][] = [
    [SIZE / 2, SIZE],
    [SIZE / 2, SIZE * 0.55],
    [SIZE * 0.78, SIZE * 0.55],
    [SIZE * 0.78, 0],
]

const canvas = createCanvas(SIZE, SIZE)
const ctx = canvas.getContext("2d")

// Warm tan background — "wooden floor" interior, tells model to fill the room
ctx.fillStyle = "#D4C4A8"
ctx.fillRect(0, 0, SIZE, SIZE)

// Floor plank grid (subtle darker lines)
ctx.strokeStyle = "#C0B090"
ctx.lineWidth = 1
for (let y = 0; y < SIZE; y += 32) {
    for (let x = 0; x < SIZE; x += 32) {
        ctx.strokeRect(x, y, 32, 32)
    }
}

// --- Room walls (thick dark lines at edges) ---
// Black walls at edges tell the model "this is the room boundary"
ctx.strokeStyle = "#000000"
ctx.lineWidth = WALL
ctx.lineCap = "square"

// Left wall
ctx.beginPath()
ctx.moveTo(0, 0)
ctx.lineTo(0, SIZE)
ctx.stroke()

// Right wall
ctx.beginPath()
ctx.moveTo(SIZE, 0)
ctx.lineTo(SIZE, SIZE)
ctx.stroke()

// Top wall (with door gap at exit point)
const exitX = route[3][0]
ctx.beginPath()
ctx.moveTo(0, 0)
ctx.lineTo(exitX - GAP, 0)
ctx.stroke()
ctx.beginPath()
ctx.moveTo(exitX + GAP, 0)
ctx.lineTo(SIZE, 0)
ctx.stroke()

// Bottom wall (with door gap at entry point)
const entryX = route[0][0]
ctx.beginPath()
ctx.moveTo(0, SIZE)
ctx.lineTo(entryX - GAP, SIZE)
ctx.stroke()
ctx.beginPath()
ctx.moveTo(entryX + GAP, SIZE)
ctx.lineTo(SIZE, SIZE)
ctx.stroke()

// --- Furniture (gray filled rectangles, lighter than walls) ---
// ctx.fillStyle = "#c0c0c0"

// // Bed in top-left
// ctx.fillRect(20, 20, 100, 60)

// // Table/candelabra top-right
// ctx.fillRect(380, 50, 80, 50)

// // Bookshelf on left wall
// ctx.fillRect(18, 220, 45, 150)

// // Fireplace on right wall
// ctx.fillRect(410, 170, 70, 100)

// // Chair bottom-left
// ctx.fillRect(50, 360, 50, 50)

// --- Walking path (medium gray line - less prominent than walls) ---
ctx.strokeStyle = "#808080"
ctx.lineWidth = PATH
ctx.lineCap = "round"
ctx.lineJoin = "round"
ctx.beginPath()
route.forEach(([x, y], i) => {
    if (i === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
})
ctx.stroke()

// Write as PNG
const buffer = canvas.toBuffer("image/png")
await Bun.write("dist/route.png", buffer)
console.log(`✅ Written dist/route.png (${SIZE}×${SIZE})`)

