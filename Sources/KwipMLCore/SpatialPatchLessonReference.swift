import Foundation

/// Independent scalar routing; expected arithmetic never calls an Unfold/Fold backend.
public struct SpatialPatchLessonReference {
    public let output:LearningTensor
    private let input:LearningTensor
    private let routes:[(source:Int,target:Int)]
    public init(input:LearningTensor,task:SpatialPatchLearningTask)throws {
        guard input.wellFormed,(1...3).contains(input.shape[0]),input.shape == [input.shape[0]]+task.sampleShape else {throw LearningContractError("Patch source must match the public grid/column shape and one to three observations.")}
        self.input=input
        func pair(_ key:String)->[Int] {if case .array(let values)=task.configuration[key] {return values.map {$0.integerValue!}};preconditionFailure("Frozen lesson geometry is a pair.")}
        let kernel=pair("kernel_size"),stride=pair("stride"),dilation=pair("dilation"),padding=pair("padding"),spatial=task.isUnfold ? Array(task.sampleShape.dropFirst()):pair("output_size")
        let h=spatial[0],w=spatial[1],kh=kernel[0],kw=kernel[1],channels=task.isUnfold ? task.sampleShape[0]:task.sampleShape[0]/(kh*kw)
        let oh=(h+2*padding[0]-dilation[0]*(kh-1)-1)/stride[0]+1,ow=(w+2*padding[1]-dilation[1]*(kw-1)-1)/stride[1]+1,columns=oh*ow,packed=channels*kh*kw
        var result=Array(repeating:0.0,count:input.shape[0]*task.outputSampleShape.reduce(1,*)),mapping:[(source:Int,target:Int)]=[]
        for n in 0..<input.shape[0] {for channel in 0..<channels {for ky in 0..<kh {for kx in 0..<kw {for py in 0..<oh {for px in 0..<ow {
            let iy=py*stride[0]-padding[0]+ky*dilation[0],ix=px*stride[1]-padding[1]+kx*dilation[1]
            guard (0..<h).contains(iy),(0..<w).contains(ix) else {continue}
            let image=((n*channels+channel)*h+iy)*w+ix,column=(n*packed+channel*kh*kw+ky*kw+kx)*columns+py*ow+px
            let source=task.isUnfold ? image:column,target=task.isUnfold ? column:image
            result[target]+=input.values[source];mapping.append((source,target))
        }}}}}}
        routes=mapping;output = .init(shape:[input.shape[0]]+task.outputSampleShape,values:result)
    }
    public func inputGradient(for upstream:LearningTensor)throws->LearningTensor {
        guard upstream.wellFormed,upstream.shape == output.shape else {throw LearningContractError("Patch output influence must match the requested result shape.")}
        var gradient=Array(repeating:0.0,count:input.values.count)
        for route in routes {gradient[route.source]+=upstream.values[route.target]}
        return .init(shape:input.shape,values:gradient)
    }
    public static let explanation="Unfold turns small rectangles into columns. Packed rows follow channel, patch row, then patch column; patch columns follow start positions in reading order. Declared border positions contain zero. A source cell used by several patches receives added influence; an unvisited cell receives none.\n\nFold routes those columns into a declared output grid and adds overlapping contributions. It does not average overlap counts or fill uncovered cells. Column entries outside the real grid do not contribute and receive no influence. Consequently Fold is not automatically the inverse of Unfold. These blocks own no learned weights or randomness. The checker uses independent scalar coordinates and CPU float64 measurements; production retains its reviewed float32/device limits."
}
