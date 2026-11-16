import Foundation

public enum ConvolutionLearningTask: String, CaseIterable, Codable {
    case one = "conv1d-shared-kernel", two = "conv2d-shared-kernel", three = "conv3d-shared-kernel"
    case transposedOne = "conv-transpose1d-shared-kernel", transposedTwo = "conv-transpose2d-shared-kernel", transposedThree = "conv-transpose3d-shared-kernel"
    public var isTransposed: Bool {[Self.transposedOne,.transposedTwo,.transposedThree].contains(self)}
    public var dimensions: Int {self == .one || self == .transposedOne ? 1 : self == .two || self == .transposedTwo ? 2 : 3}
    public var componentID: String {isTransposed ? "conv_transpose\(dimensions)d" : "conv\(dimensions)d"}
    public var title: String {isTransposed ? "ConvTranspose\(dimensions)d · overlapping output windows" : "Conv\(dimensions)d · shared local windows"}
    public var sampleShape: [Int] { [2] + Array(repeating: isTransposed ? 3 : 4, count: dimensions) }
    public var configuration: [String:JSONValue] {
        func axis(_ value: Int) -> JSONValue { dimensions == 1 ? .integer(Int64(value)) : .array(Array(repeating:.integer(Int64(value)),count:dimensions)) }
        if isTransposed {return ["in_channels":2,"out_channels":2,"kernel_size":axis(3),"stride":axis(2),"dilation":axis(1),"groups":1,"bias":true,"padding":axis(0),"output_padding":axis(1)]}
        return ["in_channels":2,"out_channels":2,"kernel_size":axis(2),"stride":axis(1),"dilation":axis(1),"groups":1,"bias":true,"padding_policy":.string("explicit"),"padding":axis(0),"padding_mode":.string("zeros")]
    }
    public var instructions: String {
        if isTransposed {return "Each observation has two channels and three positions in each spatial direction. A channel is a separate measurement field. A kernel is a shared window of weights; stride moves the start of that window, dilation spaces positions within it, padding controls boundary extent and Groups 1 connects all input channels to each output channel. Add Input, ConvTranspose\(dimensions)d and Model Output; connect them in that order. Select the transposed convolution and configure Input/Output channels 2, Kernel size 3, Stride 2, Dilation 1, Groups 1, Padding 0 and Output padding 1 in every spatial direction, with Learn bias enabled. The lesson supplies the kernel and bias automatically. Each input spreads a weighted window into output positions; overlap adds contributions instead of replacing them. Expect eight positions in each direction. Output padding chooses the extra extent; it does not add input values. The extra boundary can still contain bias even where no input window contributes. Choose Check construction to inspect output and automatic influence checks; no handwritten derivatives are required. This operation does not recover the original input."}
        return "Each observation has two channels, with four positions in each of its \(dimensions) spatial directions. A channel is a separate measurement field; a spatial direction follows positions along a line, across a table or through a volume. Add Input, Conv\(dimensions)d and Model Output, and connect them in that order. Select Conv\(dimensions)d to configure it. Set Input channels and Output channels to 2, Kernel size to 2 in every spatial direction, Stride and Dilation to 1, Groups to 1, and enable Learn bias. Choose Explicit zero padding with amount 0 (Valid with amount 0 is equivalent). The lesson supplies the shared kernel weights and bias values automatically. Choose Check construction to inspect the measured output and influence on the input and shared weights; no code or handwritten derivatives are needed. The same small window moves across each observation, keeping channels and observations distinct. Expect two output channels and three positions in each spatial direction."
    }
}
public enum ConvolutionAssessment {
    public static func plan(task:ConvolutionLearningTask,runtimeVersion:String,seed:UInt64)throws->LearningAssessmentPlan {
        if task.isTransposed {return try transposedPlan(task:task,runtimeVersion:runtimeVersion,seed:seed)}
        var state=seed
        func tensor(_ shape:[Int])->LearningTensor { .init(shape:shape,values:(0..<shape.reduce(1,*)).map { _ in state=state &* 6364136223846793005 &+ 1442695040888963407; return Double(Int((state >> 32)%17)-8)/4 }) }
        let fixtures=try (0..<4).map { i -> LearningFixture in
            let input=tensor([i%2+1]+task.sampleShape),kernel=tensor([2,2]+Array(repeating:2,count:task.dimensions)),bias=tensor([2])
            let reference=try ConvolutionLessonReference(input:input,kernel:kernel,bias:bias,dimensions:task.dimensions)
            let upstream=tensor(reference.output.shape),influence=try reference.gradients(for:upstream)
            return .init(id:"\(task.rawValue)-\(i)",inputs:["x":input,"weight":kernel,"bias":bias,"upstream":upstream],expected:["output":reference.output,"inputGradient":influence.input,"kernelGradient":influence.kernel,"biasGradient":influence.bias])
        }
        let descriptions=[("output","Move the shared window across each observation and combine its channels."),("inputGradient","Check how small input changes affect all windows using them."),("kernelGradient","Combine the influence on shared weights across their uses."),("biasGradient","Combine the influence on each shared output offset.")]
        let plan=LearningAssessmentPlan(id:"assessment."+task.rawValue,objectiveID:"convolution."+task.rawValue,task:task.instructions,runtime:.init(backend:"pytorch",version:runtimeVersion,capabilities:[task.componentID:ContractVersion()]),fixtureSeed:seed,fixtures:fixtures,criteria:descriptions.map {.init(id:$0.0,description:$0.1,nextStep:"Check the channel counts, shared Kernel size 2, Stride and Dilation 1, Groups 1, zero padding and enabled bias. Connect the supplied observations through the requested convolution to Model Output, then measure again.",rule:.tensor(key:$0.0,absoluteTolerance:1e-6,relativeTolerance:1e-6))})
        try plan.validate();return plan
    }
    private static func transposedPlan(task:ConvolutionLearningTask,runtimeVersion:String,seed:UInt64)throws->LearningAssessmentPlan {
        var state=seed
        func tensor(_ shape:[Int])->LearningTensor {.init(shape:shape,values:(0..<shape.reduce(1,*)).map {_ in state=state &* 6364136223846793005 &+ 1442695040888963407;return Double(Int((state >> 32)%17)-8)/4})}
        let fixtures=try (0..<4).map {i -> LearningFixture in
            let input=tensor([i%2+1]+task.sampleShape),kernel=tensor([2,2]+Array(repeating:3,count:task.dimensions)),bias=tensor([2])
            let reference=try TransposedConvolutionLessonReference(input:input,kernel:kernel,bias:bias,dimensions:task.dimensions)
            let upstream=tensor(reference.output.shape),influence=try reference.gradients(for:upstream)
            return .init(id:"\(task.rawValue)-\(i)",inputs:["x":input,"weight":kernel,"bias":bias,"upstream":upstream],expected:["output":reference.output,"inputGradient":influence.input,"kernelGradient":influence.kernel,"biasGradient":influence.bias])
        }
        let descriptions=[("output","Add overlapping output-window contributions, then each channel’s bias."),("inputGradient","Measure how input changes affect scattered output windows."),("kernelGradient","Accumulate influence on the shared scatter weights."),("biasGradient","Include every output position in the shared offset influence.")]
        let plan=LearningAssessmentPlan(id:"assessment."+task.rawValue,objectiveID:"convolution."+task.rawValue,task:task.instructions,runtime:.init(backend:"pytorch",version:runtimeVersion,capabilities:[task.componentID:ContractVersion()]),budget:.init(milliseconds:5000,maximumTensorElements:16384),fixtureSeed:seed,fixtures:fixtures,criteria:descriptions.map {.init(id:$0.0,description:$0.1,nextStep:"Check two channels, Kernel size 3, Stride 2, Dilation 1, Groups 1, Padding 0, Output padding 1 and enabled bias. Overlapping windows add contributions; output padding changes extent, not input data. Choose Check construction again.",rule:.tensor(key:$0.0,absoluteTolerance:1e-6,relativeTolerance:1e-6))})
        try plan.validate();return plan
    }
}
