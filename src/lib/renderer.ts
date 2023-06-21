import vertexShader from '../shaders/vertex.wgsl'
import fragmentShader from '../shaders/fragment.wgsl'
import computeShader from '../shaders/compute.wgsl'
import {InfoInterface} from "../index";

function getTime() {
    return (new Date()).getMilliseconds()
}

function getRandomValue(v1: number, v2 = 0) {
    const max = Math.max(v1, v2)
    const min = Math.min(v1, v2)
    return Math.random() * (max - min) + min;
}

function getInRange(range: [number, number]) {
    return getRandomValue(...range)
}

function radians(angle: number) {
    return angle / 180 * Math.PI
}

function HSLToRGB(h: number, s: number, l: number): [number, number, number] {
    s /= 100;
    l /= 100;
    const k = (n: number) => (n + h / 30) % 12;
    const a = s * Math.min(l, 1 - l);
    const f = (n: number) => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
    return [f(0), f(8), f(4)];
}

export default class {
    canvas: HTMLCanvasElement;
    info: InfoInterface
    step: number
    resolution: [number, number]
    gridResolution: [number, number, number]
    circleCount: number
    deltaTime: number
    circleMaximumRadius: number
    circleParams: number
    gridCellParams: number
    workgroupSize: number
    workgroupUpdateCirclesCount: number
    workgroupSolveCollisionsCount: [number, number]

    // API Data Structures
    adapter: GPUAdapter;
    device: GPUDevice;
    queue: GPUQueue;

    // Frame Backings
    context: GPUCanvasContext;
    canvasFormat: GPUTextureFormat;

    // Arrays
    vertexArray: Float32Array
    timeArray: Uint32Array
    resolutionArray: Float32Array
    gridResolutionArray: Uint32Array
    gridArray: Uint32Array
    circlesArray: Float32Array
    globalParamsArray: Float32Array

    // Buffers
    vertexBuffer: GPUBuffer
    timeBuffer: GPUBuffer
    resolutionBuffer: GPUBuffer
    gridResolutionBuffer: GPUBuffer
    gridBuffer: GPUBuffer
    circlesBuffer: GPUBuffer
    globalParamsBuffer: GPUBuffer

    // Layouts
    vertexBufferLayout: GPUVertexBufferLayout
    bindGroupComputeLayout: GPUBindGroupLayout
    bindGroupRenderLayout: GPUBindGroupLayout
    pipelineComputeLayout: GPUPipelineLayout
    pipelineRenderLayout: GPUPipelineLayout

    // Bind groups
    bindGroupCompute: GPUBindGroup
    bindGroupRender: GPUBindGroup

    // Pipelines
    solveCollisionsPipeline: GPUComputePipeline
    updateCirclesPipeline: GPUComputePipeline
    renderPipeline: GPURenderPipeline

    constructor(canvas: HTMLCanvasElement, info: InfoInterface) {
        this.canvas = canvas
        this.info = info

        this.resolution = [canvas.width, canvas.height];
        this.circleCount = 4
        this.deltaTime = 1
        this.circleMaximumRadius = 12
        this.circleParams = 12
        this.gridCellParams = 10

        this.step = 0
        this.workgroupSize = 8;
        this.gridResolution = [
            Math.ceil(this.resolution[0] / this.circleMaximumRadius),
            Math.ceil(this.resolution[1] / this.circleMaximumRadius),
            this.gridCellParams
        ]

        this.workgroupSolveCollisionsCount = [
            Math.ceil(this.gridResolution[0] / this.workgroupSize),
            Math.ceil(this.gridResolution[1] / this.workgroupSize)
        ];
        this.workgroupUpdateCirclesCount = Math.ceil(this.circleCount / (this.workgroupSize * this.workgroupSize))
    }

    update() {
        const t = getTime()
        this.step++
        this.timeArray[0] = this.step;
        this.writeBuffer(this.timeBuffer, this.timeArray)
        this.globalParamsArray[0] = this.step;
        this.writeBuffer(this.globalParamsBuffer, this.globalParamsArray)

        const encoder = this.device.createCommandEncoder();
        this.updateCircles(encoder)
        this.solveCollisions(encoder)
        this.render(encoder)
        this.queue.submit([encoder.finish()]);

        const dt = getTime() - t
        this.info.renderTime.innerText = `${dt} ms`
        requestAnimationFrame(() => this.update())
    }

    initCircles() {
        for (let i = 0; i < this.circleCount; i++) {
            const ind = i * this.circleParams
            // 0 position
            this.circlesArray[ind] = this.resolution[0] / 2 + Math.cos(radians(i * 80)) * 100
            this.circlesArray[ind + 1] = this.resolution[1] / 2 + Math.sin(radians(i * 80)) * 100

            // 8 last position
            this.circlesArray[ind + 2] = this.circlesArray[ind] - Math.cos(radians(i * 80)) * 2
            this.circlesArray[ind + 3] = this.circlesArray[ind + 1] - Math.sin(radians(i * 80)) * 2

            // 16 acceleration
            this.circlesArray[ind + 4] = 0
            this.circlesArray[ind + 5] = -0.1

            // 24 alignment
            this.circlesArray[ind + 6] = 0
            this.circlesArray[ind + 7] = 0

            // 32 color
            const color = HSLToRGB(i * 80, 100, 50)
            this.circlesArray[ind + 8] = color[0]
            this.circlesArray[ind + 9] = color[1]
            this.circlesArray[ind + 10] = color[2]

            // 44 radius
            this.circlesArray[ind + 11] = getInRange([8, 8])

            // 48
        }
    }

    updateCircles(encoder: GPUCommandEncoder) {
        const computePass = encoder.beginComputePass();
        computePass.setPipeline(this.updateCirclesPipeline)
        computePass.setBindGroup(0, this.bindGroupCompute);
        computePass.dispatchWorkgroups(this.workgroupUpdateCirclesCount);
        computePass.end();
    }

    solveCollisions(encoder: GPUCommandEncoder) {
        const computePass = encoder.beginComputePass();
        computePass.setPipeline(this.solveCollisionsPipeline)
        computePass.setBindGroup(0, this.bindGroupCompute);
        computePass.dispatchWorkgroups(this.workgroupSolveCollisionsCount[0], this.workgroupSolveCollisionsCount[1]);
        computePass.end();
    }

    render(encoder: GPUCommandEncoder) {
        const pass = encoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                loadOp: "clear",
                clearValue: {r: 0, g: 0, b: 0, a: 1.0},
                storeOp: "store",
            }]
        });
        pass.setPipeline(this.renderPipeline);
        pass.setBindGroup(0, this.bindGroupRender)
        pass.setVertexBuffer(0, this.vertexBuffer);
        pass.draw(this.vertexArray.length / 2, Math.min(this.circleCount, this.step));
        pass.end();
    }

    async init() {
        if (await this.initApi()) {
            console.log(this.resolution, this.circleCount)
            this.initCanvas()
            this.createArrays()
            this.createBuffers()
            this.writeBuffers()
            this.createLayouts()
            this.createBindings()
            this.createPipelines()
            return true
        } else {
            return false
        }
    }

    createArrays() {
        this.vertexArray = new Float32Array([
            -1, -1, 1, -1, 1, 1,
            -1, -1, 1, 1, -1, 1,
        ]);

        this.timeArray = new Uint32Array([0]);
        this.resolutionArray = new Float32Array(this.resolution);
        this.gridResolutionArray = new Uint32Array(this.gridResolution);
        this.gridArray = new Uint32Array(this.gridResolutionArray[0] * this.gridResolutionArray[1] * this.gridCellParams);
        this.circlesArray = new Float32Array(this.circleCount * this.circleParams);
        this.globalParamsArray = new Float32Array([
            this.circleCount,
            this.deltaTime
        ])
        this.initCircles()
    }

    createBuffers() {
        this.vertexBuffer = this.createBuffer('vertices', this.vertexArray, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST)
        this.resolutionBuffer = this.createBuffer('resolution', this.resolutionArray, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        this.timeBuffer = this.createBuffer('time', this.timeArray, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        this.gridResolutionBuffer = this.createBuffer('grid resolution', this.gridResolutionArray, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        this.globalParamsBuffer = this.createBuffer('global params', this.globalParamsArray, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)
        this.circlesBuffer = this.createBuffer('circles', this.circlesArray, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST)
        this.gridBuffer = this.createBuffer('grid', this.gridArray, GPUBufferUsage.STORAGE)
    }

    writeBuffers() {
        this.writeBuffer(this.vertexBuffer, this.vertexArray)
        this.writeBuffer(this.timeBuffer, this.timeArray)
        this.writeBuffer(this.resolutionBuffer, this.resolutionArray)
        this.writeBuffer(this.gridResolutionBuffer, this.gridResolutionArray)
        this.writeBuffer(this.circlesBuffer, this.circlesArray)
        this.writeBuffer(this.globalParamsBuffer, this.globalParamsArray)
    }

    createLayouts() {
        this.vertexBufferLayout = this.createVertexLayout(this.vertexArray.BYTES_PER_ELEMENT * 2, 'float32x2')
        this.bindGroupComputeLayout = this.device.createBindGroupLayout({
            label: "bind group compute layout",
            entries: [{
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {type: "uniform"}
            }, {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {type: "uniform"}
            }, {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {type: "uniform"}
            }, {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {type: "uniform"}
            }, {
                binding: 4,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {type: "storage"}
            }, {
                binding: 5,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {type: "storage"}
            }]
        });

        this.pipelineComputeLayout = this.device.createPipelineLayout({
            label: "compute pipeline layout",
            bindGroupLayouts: [this.bindGroupComputeLayout],
        });

        this.bindGroupRenderLayout = this.device.createBindGroupLayout({
            label: "bind group render layout",
            entries: [{
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: {type: "read-only-storage"}
            }, {
                binding: 1,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: {type: "uniform"}
            }, {
                binding: 2,
                visibility: GPUShaderStage.FRAGMENT,
                buffer: {type: "uniform"}
            }]
        });

        this.pipelineRenderLayout = this.device.createPipelineLayout({
            label: "render pipeline layout",
            bindGroupLayouts: [this.bindGroupRenderLayout],
        });
    }

    createBindings() {
        this.bindGroupCompute = this.device.createBindGroup({
            label: "Bind group compute",
            layout: this.bindGroupComputeLayout,
            entries: [{
                binding: 0,
                resource: {buffer: this.resolutionBuffer}
            }, {
                binding: 1,
                resource: {buffer: this.timeBuffer}
            }, {
                binding: 2,
                resource: {buffer: this.gridResolutionBuffer}
            }, {
                binding: 3,
                resource: {buffer: this.globalParamsBuffer}
            }, {
                binding: 4,
                resource: {buffer: this.circlesBuffer}
            }, {
                binding: 5,
                resource: {buffer: this.gridBuffer}
            }],
        })

        this.bindGroupRender = this.device.createBindGroup({
            label: "Bind group compute",
            layout: this.bindGroupRenderLayout,
            entries: [{
                binding: 0,
                resource: {buffer: this.circlesBuffer}
            }, {
                binding: 1,
                resource: {buffer: this.resolutionBuffer}
            }, {
                binding: 2,
                resource: {buffer: this.globalParamsBuffer}
            }]
        })
    }

    createPipelines() {
        const fragmentModule = this.device.createShaderModule({code: fragmentShader});
        const vertexModule = this.device.createShaderModule({code: vertexShader});
        const computeModule = this.device.createShaderModule({code: computeShader});

        this.updateCirclesPipeline = this.device.createComputePipeline({
            label: "update circles pipeline",
            layout: this.pipelineComputeLayout,
            compute: {
                module: computeModule,
                entryPoint: "update_circles",
            }
        });

        this.solveCollisionsPipeline = this.device.createComputePipeline({
            label: "solve collisions pipeline",
            layout: this.pipelineComputeLayout,
            compute: {
                module: computeModule,
                entryPoint: "solve_collisions",
            }
        });

        this.renderPipeline = this.device.createRenderPipeline({
            label: "render pipeline",
            layout: this.pipelineRenderLayout,
            vertex: {
                module: vertexModule,
                entryPoint: "main",
                buffers: [this.vertexBufferLayout]
            },
            fragment: {
                module: fragmentModule,
                entryPoint: "main",
                targets: [{
                    format: this.canvasFormat
                }]
            }
        });
    }

    async initApi() {
        try {
            this.adapter = await navigator.gpu.requestAdapter();
            this.device = await this.adapter.requestDevice();
            this.queue = this.device.queue
            console.log('Adapter: ', this.adapter)
            console.log('Device: ', this.device)
        } catch (e) {
            console.log(e)
            return false
        }
        return true
    }

    initCanvas() {
        this.context = this.canvas.getContext("webgpu");
        this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: this.device,
            format: this.canvasFormat,
        });
    }

    createBuffer(label: string, array: BufferSource, usage: GPUBufferUsageFlags) {
        return this.device.createBuffer({
            label: label,
            size: array.byteLength,
            usage: usage,
        });
    }

    writeBuffer(gpuBuffer: GPUBuffer, data: BufferSource | SharedArrayBuffer) {
        this.queue.writeBuffer(gpuBuffer, 0, data);
    }

    createVertexLayout(arrayStride: number, format: GPUVertexFormat): GPUVertexBufferLayout {
        return {
            arrayStride: arrayStride,
            attributes: [{
                format: format,
                offset: 0,
                shaderLocation: 0, // Position, see vertex shader
            }],
        };
    }
}