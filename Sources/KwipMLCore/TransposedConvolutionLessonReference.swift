import Foundation

/// Independent scatter-add oracle; no convolution library supplies expected values.
public struct TransposedConvolutionLessonReference {
    public let output:LearningTensor
    private let input:LearningTensor,kernel:LearningTensor,bias:LearningTensor
    private let dimensions:Int
    public init(input:LearningTensor,kernel:LearningTensor,bias:LearningTensor,dimensions:Int)throws {
        guard (1...3).contains(dimensions),input.wellFormed,kernel.wellFormed,bias.wellFormed,
              input.shape.count == dimensions+2,input.shape[0]<=2,Array(input.shape.dropFirst()) == [2]+Array(repeating:3,count:dimensions),
              kernel.shape == [2,2]+Array(repeating:3,count:dimensions),bias.shape == [2] else {throw LearningContractError("Scatter-window reference requires two channels, spatial sizes3, kernel sizes3 and two offsets.")}
        self.input=input;self.kernel=kernel;self.bias=bias;self.dimensions=dimensions
        let shape=[input.shape[0],2]+Array(repeating:8,count:dimensions),spatial=Int(pow(8.0,Double(dimensions)))
        var values=(0..<shape.reduce(1,*)).map {bias.values[($0/spatial)%2]}
        Self.visit(shape:input.shape,dimensions:dimensions) {out,source,weight in values[out]+=input.values[source]*kernel.values[weight]}
        output = .init(shape:shape,values:values)
    }
    private static func coordinates(_ flat:Int,dimensions:Int)->[Int] {var value=flat;var result=Array(repeating:0,count:dimensions);for d in (0..<dimensions).reversed() {result[d]=value%3;value/=3};return result}
    private static func visit(shape:[Int],dimensions:Int,body:(Int,Int,Int)->Void) {
        let spatial=Int(pow(3.0,Double(dimensions))),outputSpatial=Int(pow(8.0,Double(dimensions)))
        for n in 0..<shape[0] {for channel in 0..<2 {for p in 0..<spatial {
            let position=coordinates(p,dimensions:dimensions),source=(n*2+channel)*spatial+p
            for outChannel in 0..<2 {for k in 0..<spatial {
                let offset=coordinates(k,dimensions:dimensions);var destination=0
                for d in 0..<dimensions {destination=destination*8+position[d]*2+offset[d]}
                body((n*2+outChannel)*outputSpatial+destination,source,(channel*2+outChannel)*spatial+k)
            }}
        }}}
    }
    public func gradients(for upstream:LearningTensor)throws->(input:LearningTensor,kernel:LearningTensor,bias:LearningTensor) {
        guard upstream.wellFormed,upstream.shape == output.shape else {throw LearningContractError("Scatter-window influence must match the output shape.")}
        var dx=Array(repeating:0.0,count:input.values.count),dw=Array(repeating:0.0,count:kernel.values.count),db=[0.0,0.0]
        Self.visit(shape:input.shape,dimensions:dimensions) {out,source,weight in dx[source]+=upstream.values[out]*kernel.values[weight];dw[weight]+=upstream.values[out]*input.values[source]}
        let spatial=Int(pow(8.0,Double(dimensions)));for i in upstream.values.indices {db[(i/spatial)%2]+=upstream.values[i]}
        return (.init(shape:input.shape,values:dx),.init(shape:kernel.shape,values:dw),.init(shape:bias.shape,values:db))
    }
    public static let explanation="A transposed convolution scatters each input value through a shared kernel into output positions, adding contributions where windows overlap. X names the input, W the weights, b the output offset and Y the result. With stride2, input position p begins its window at 2p; kernel position k contributes to output position 2p+k. The shared weight layout starts with input channels, then output channels, then spatial directions. This transposes the connectivity of a convolution; it does not recover the original input. Output padding chooses an additional output extent, not extra input data. Here three input positions, stride2 and kernel3 cover positions0–6; output padding1 extends to position7. A position receiving no kernel contribution still contains its channel’s bias. Influence on inputs and weights adds contributions from every use; bias influence includes every output position. The lesson checks float64 scalar arithmetic independently of the production float32 contract."
}
