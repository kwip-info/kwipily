import Foundation

/// Independent gather/scatter teaching oracle with exact integer coordinates.
/// Indices are discrete selectors: there is no differentiable index gradient.
public struct EmbeddingLessonReference {
    public static let explanation = "W is the shared lookup table, V is its number of rows, and D is the number of values in each row. W ∈ ℝ^(V×D) means a V-by-D table of real numbers. N is the number of observations. An index i is a whole-number row label: output position p and coordinate c contain y[p,c] = W[I[p],c], where I is the array of row selectors. Each selected row is a vector, an ordered list of values. Selector shape [N,…] becomes [N,…,D]; the dots stand for any remaining grouping directions. Selecting a row is also equivalent to a one-hot vector times W, where a one-hot vector has 1 at the selected row and 0 elsewhere; the lookup does not need to allocate it. A gradient records how small changes to the learned table affect a combined result. L is that combined result, y is the lookup output, p selects an output position and c selects a coordinate within its vector. The notation dL/dW names that table gradient, and an upstream contribution is the influence supplied for each output coordinate. Contributions add into the selected rows; repeated selections accumulate because they share the same table values. Unselected rows have zero contribution. Padding suppression and division by selection frequency are explicit gradient policies. Norm means vector length: order 2 uses the square root of the sum of squared coordinates. Maximum norm changes only selected over-limit rows before lookup, using W ← W/(norm(W)+1e−7); the small added amount is the runtime numerical safeguard. This forward change is outside differentiation and still occurs for frozen weights. Freeze stops learned gradient updates; it does not suppress this configured forward change. Discrete category indices are labels and have no differentiable gradient. These equations explain automatic checks, not answers the learner must type."

    public let output:LearningTensor
    public let tableAfterForward:LearningTensor
    private let indices:[Int]
    private let configuration:EmbeddingConfiguration

    public init(indices:[Int64],shape:[Int],table:LearningTensor,configuration:EmbeddingConfiguration)throws {
        guard table.wellFormed,table.shape == [configuration.numberOfEmbeddings,configuration.embeddingDimension],
              (2...7).contains(shape.count),!indices.isEmpty,indices.count<=4096,
              !(configuration.sparse && configuration.scaleGradientByFrequency) else {
            throw LearningContractError("Lookup requires a bounded table, batched integer coordinates and compatible gradient settings.")
        }
        var count=1
        for size in shape {
            guard (1...4096).contains(size),count<=4096/size else {throw LearningContractError("Index shape exceeds the lesson budget.")}
            count*=size
        }
        guard count==indices.count,count<=4096/configuration.embeddingDimension,
              indices.allSatisfy({$0>=0 && $0<Int64(configuration.numberOfEmbeddings)}) else {
            throw LearningContractError("Every index must select an existing table row; shape and output must fit the lesson budget.")
        }
        self.indices=indices.map(Int.init);self.configuration=configuration
        let width=configuration.embeddingDimension
        var weights=table.values
        if let maximum=configuration.maximumNorm {
            // Renormalization is a no-grad forward mutation. It is not part of
            // the gather backward derivative, even when parameters are frozen.
            for row in Set(self.indices).sorted() {
                let start=row*width
                let largest=(0..<width).map {abs(weights[start+$0])}.max() ?? 0
                let norm=largest==0 ? 0 : largest*pow((0..<width).reduce(0.0) {$0+pow(abs(weights[start+$1])/largest,configuration.normType)},1/configuration.normType)
                guard norm.isFinite else {throw LearningContractError("Table row norm exceeded finite reference range.")}
                if norm>maximum {
                    let scale=maximum/(norm+1e-7)
                    for column in 0..<width {weights[start+column]*=scale}
                }
            }
        }
        tableAfterForward=LearningTensor(shape:table.shape,values:weights)
        output=LearningTensor(shape:shape+[width],values:self.indices.flatMap {row in Array(weights[(row*width)..<(row*width+width)])})
    }

    /// Padding suppression and frequency scaling are explicit backward policies;
    /// they need not equal finite differences of the unmodified gather function.
    /// Dense mathematical representation of the table gradient, including when
    /// the execution backend stores it sparsely. Nil means frozen parameters.
    public func tableGradient(for upstream:LearningTensor)throws->LearningTensor? {
        guard upstream.wellFormed,upstream.shape==output.shape else {throw LearningContractError("Lookup upstream gradients must match the complete output shape.")}
        guard !configuration.frozen else {return nil}
        let width=configuration.embeddingDimension
        var frequency:[Int:Int]=[:]
        for row in indices {frequency[row,default:0]+=1}
        var values=Array(repeating:0.0,count:tableAfterForward.values.count)
        for (position,row) in indices.enumerated() where row != configuration.paddingIndex {
            let divisor=configuration.scaleGradientByFrequency ? Double(frequency[row]!) : 1
            for column in 0..<width {values[row*width+column]+=upstream.values[position*width+column]/divisor}
        }
        guard values.allSatisfy(\.isFinite) else {throw LearningContractError("Lookup gradient accumulation exceeded finite reference range.")}
        return LearningTensor(shape:tableAfterForward.shape,values:values)
    }
}
