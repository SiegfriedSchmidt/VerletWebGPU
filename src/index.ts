import "./styles/main.css"
import Renderer from "./lib/renderer";

export interface InfoInterface {
    renderTime: HTMLParagraphElement
}

const infoRenderTime = document.getElementById('renderTime') as HTMLParagraphElement
const info: InfoInterface = {renderTime: infoRenderTime}

const canvas = document.getElementById('root') as HTMLCanvasElement
const renderer = new Renderer(canvas, info)
if (await renderer.init()) {
    renderer.update()
} else {
    document.body.innerHTML = '<div class="not-supported"><h1>WebGPU not supported!</h1></div>'
}
