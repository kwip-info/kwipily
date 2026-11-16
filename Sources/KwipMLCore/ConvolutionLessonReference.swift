import Foundation

/// Independent scalar cross-correlation for the bounded shared-window objective.
public struct ConvolutionLessonReference {
    public let output:LearningTensor
    private let input:LearningTensor, kernel:LearningTensor, bias:LearningTensor
    private let dimensions:Int
    public init(input:LearningTensor,kernel:LearningTensor,bias:LearningTensor,dimensions:Int)throws {
        guard (1...3).contains(dimensions),input.wellFormed,kernel.wellFormed,bias.wellFormed,
              input.shape.count == dimensions+2,kernel.shape == [2,2]+Array(repeating:2,count:dimensions),
              Array(input.shape.dropFirst()) == [2]+Array(repeating:4,count:dimensions),bias.shape == [2],input.shape[0]<=2 else {throw LearningContractError("Shared-window reference requires two channels, spatial sizes 4, kernel sizes 2 and two offsets.")}
        self.input=input;self.kernel=kernel;self.bias=bias;self.dimensions=dimensions
        let outputShape=[input.shape[0],2]+Array(repeating:3,count:dimensions)
        var values=Array(repeating:0.0,count:outputShape.reduce(1,*))
        Self.visit(inputShape:input.shape,dimensions:dimensions) { out,source,weight,channel in values[out]+=input.values[source]*kernel.values[weight] }
        let spatial=Int(pow(3.0,Double(dimensions)))
        for i in values.indices {values[i]+=bias.values[(i/spatial)%2]}
        output = .init(shape:outputShape,values:values)
    }
    private static func coordinates(_ flat:Int,size:Int,dimensions:Int)->[Int] { var value=flat;var result=Array(repeating:0,count:dimensions);for d in (0..<dimensions).reversed() {result[d]=value%size;value/=size};return result }
    private static func visit(inputShape:[Int],dimensions:Int,body:(Int,Int,Int,Int)->Void) {
        let inputSpatial=Int(pow(4.0,Double(dimensions))),outputSpatial=Int(pow(3.0,Double(dimensions))),kernelSpatial=Int(pow(2.0,Double(dimensions)))
        for n in 0..<inputShape[0] {for outChannel in 0..<2 {for position in 0..<outputSpatial {
            let location=coordinates(position,size:3,dimensions:dimensions),out=(n*2+outChannel)*outputSpatial+position
            for channel in 0..<2 {for k in 0..<kernelSpatial {
                let offset=coordinates(k,size:2,dimensions:dimensions)
                var cell=0;for axis in 0..<dimensions {cell=cell*4+location[axis]+offset[axis]}
                body(out,(n*2+channel)*inputSpatial+cell,(outChannel*2+channel)*kernelSpatial+k,outChannel)
            }}
        }}}
    }
    public func gradients(for upstream:LearningTensor)throws->(input:LearningTensor,kernel:LearningTensor,bias:LearningTensor) {
        guard upstream.wellFormed,upstream.shape == output.shape else {throw LearningContractError("The supplied influence must match the shared-window output.")}
        var dx=Array(repeating:0.0,count:input.values.count),dw=Array(repeating:0.0,count:kernel.values.count),db=[0.0,0.0]
        Self.visit(inputShape:input.shape,dimensions:dimensions) {out,source,weight,_ in dx[source]+=upstream.values[out]*kernel.values[weight];dw[weight]+=upstream.values[out]*input.values[source]}
        let spatial=Int(pow(3.0,Double(dimensions)));for i in upstream.values.indices {db[(i/spatial)%2]+=upstream.values[i]}
        return (.init(shape:input.shape,values:dx),.init(shape:kernel.shape,values:dw),.init(shape:bias.shape,values:db))
    }
    public static let explanation="A kernel is a small set of shared weights moved over spatial positions. X names the input, W the kernel weights, b the output offset and Y the result. At each position, multiply matching input and kernel entries and add them across channels and window positions, then add b. The kernel is not reversed: this operation is cross-correlation, the convention used by these convolution blocks. A gradient records how a small change affects a combined result. Each input position collects influence from every overlapping window that uses it; each shared weight collects influence from every observation and position using it. Offsets collect contributions for their output channel. Stride is the step between windows, dilation the spacing between positions inside a window, and padding adds boundary positions. This finite lesson uses stride/dilation 1 and no padding. Scalar checks use double-precision lesson arithmetic; production model execution remains its reviewed float32 contract."
}
