import Foundation

/// Independent row-loop oracle for the six bounded lessons; no native adapter.
/// Source rows are table cells, not a fabricated fixed-shape packed model input.
public struct EmbeddingBagLessonReference {
    public static let explanation="A bag is one observation's group of category IDs. An ID selects a row in the shared table; that row is a vector, an ordered list of numbers. Sum adds the selected vectors, mean averages them, and max keeps the largest value separately in each position. An offset marks where a group starts in a packed list. Empty bags keep their observation row and return zeros. Padding is excluded from the combination. A weight controls an item's influence and is available with sum. Gradients measure how small changes affect the result: table values and real item weights can have gradients, but whole-number IDs and group offsets cannot."
    public let output:LearningTensor
    public let tableAfterForward:LearningTensor
    private let rows:LearningTensor
    private let lengths:[Int]
    private let itemWeights:LearningTensor?
    private let configuration:EmbeddingBagConfiguration

    public init(sourceRows:LearningTensor,lengths:[Int],table:LearningTensor,itemWeights:LearningTensor?,configuration:EmbeddingBagConfiguration)throws {
        let c=configuration.embedding
        guard sourceRows.wellFormed,sourceRows.shape.count == 2,sourceRows.shape[1]<=128,
              table.wellFormed,table.shape == [c.numberOfEmbeddings,c.embeddingDimension],
              lengths.count == sourceRows.shape[0],lengths.allSatisfy({(0...sourceRows.shape[1]).contains($0)}),
              configuration.layout != "fixed" || lengths.allSatisfy({$0 == sourceRows.shape[1]}),
              sourceRows.shape[0]<=4096/c.embeddingDimension,
              sourceRows.values.allSatisfy({$0.rounded(.towardZero) == $0 && $0>=0 && $0<Double(c.numberOfEmbeddings)}),
              configuration.usesPerSampleWeights == (itemWeights != nil),
              !c.scaleGradientByFrequency else {throw LearningContractError("Use bounded source rows, exact category IDs and valid lengths. These six reference lessons keep frequency scaling disabled.")}
        if let weights=itemWeights {guard weights.wellFormed,weights.shape == sourceRows.shape else {throw LearningContractError("Each source ID cell needs its matching finite weight cell.")}}
        rows=sourceRows;self.lengths=lengths;self.itemWeights=itemWeights;self.configuration=configuration
        let width=c.embeddingDimension,stride=sourceRows.shape[1]
        var weights=table.values
        var selected=Set<Int>()
        for b in lengths.indices {for j in 0..<lengths[b] {selected.insert(Int(sourceRows.values[b*stride+j]))}}
        if let maximum=c.maximumNorm {
            for row in selected.sorted() {
                let start=row*width
                let largest=(0..<width).map {abs(weights[start+$0])}.max() ?? 0
                let norm=largest == 0 ? 0:largest*pow((0..<width).reduce(0.0){$0+pow(abs(weights[start+$1])/largest,c.normType)},1/c.normType)
                guard norm.isFinite else {throw LearningContractError("Selected row length must remain finite.")}
                if norm>maximum {for column in 0..<width {weights[start+column]*=maximum/(norm+1e-7)}}
            }
        }
        tableAfterForward=LearningTensor(shape:table.shape,values:weights)
        var result=Array(repeating:0.0,count:lengths.count*width)
        for b in lengths.indices {
            let positions=(0..<lengths[b]).map {b*stride+$0}.filter {Int(sourceRows.values[$0]) != c.paddingIndex}
            for column in 0..<width {
                if configuration.mode == "max" {result[b*width+column]=positions.map {weights[Int(sourceRows.values[$0])*width+column]}.max() ?? 0}
                else {
                    let sum=positions.reduce(0.0){$0+weights[Int(sourceRows.values[$1])*width+column]*(itemWeights?.values[$1] ?? 1)}
                    result[b*width+column]=sum/(configuration.mode == "mean" ? Double(max(1,positions.count)):1)
                }
            }
        }
        guard result.allSatisfy(\.isFinite) else {throw LearningContractError("Bag combination must remain finite.")}
        output=LearningTensor(shape:[lengths.count,width],values:result)
    }
    public func tableGradient(for upstream:LearningTensor)throws->LearningTensor? {
        try check(upstream)
        let c=configuration.embedding
        guard !c.frozen else {return nil}
        let width=c.embeddingDimension,stride=rows.shape[1]
        var gradient=Array(repeating:0.0,count:tableAfterForward.values.count)
        for b in lengths.indices {
            let positions=(0..<lengths[b]).map {b*stride+$0}.filter {Int(rows.values[$0]) != c.paddingIndex}
            for column in 0..<width {
                if configuration.mode == "max" {
                    if let winner=positions.first(where:{tableAfterForward.values[Int(rows.values[$0])*width+column] == output.values[b*width+column]}) {gradient[Int(rows.values[winner])*width+column]+=upstream.values[b*width+column]}
                } else {
                    for p in positions {gradient[Int(rows.values[p])*width+column]+=upstream.values[b*width+column]*(itemWeights?.values[p] ?? 1)/(configuration.mode == "mean" ? Double(max(1,positions.count)):1)}
                }
            }
        }
        guard gradient.allSatisfy(\.isFinite) else {throw LearningContractError("Table influence must remain finite.")}
        return LearningTensor(shape:tableAfterForward.shape,values:gradient)
    }
    public func itemWeightGradient(for upstream:LearningTensor)throws->LearningTensor? {
        try check(upstream)
        guard itemWeights != nil else {return nil}
        let c=configuration.embedding,width=c.embeddingDimension,stride=rows.shape[1]
        var gradient=Array(repeating:0.0,count:rows.values.count)
        for b in lengths.indices {for j in 0..<lengths[b] {
            let p=b*stride+j,row=Int(rows.values[p])
            if row != c.paddingIndex {gradient[p]=(0..<width).reduce(0.0){$0+upstream.values[b*width+$1]*tableAfterForward.values[row*width+$1]}}
        }}
        guard gradient.allSatisfy(\.isFinite) else {throw LearningContractError("Item-weight influence must remain finite.")}
        return LearningTensor(shape:rows.shape,values:gradient)
    }
    private func check(_ upstream:LearningTensor)throws {guard upstream.wellFormed,upstream.shape == output.shape else {throw LearningContractError("Influence rows must match all bag outputs, including empty bags.")}}
}
