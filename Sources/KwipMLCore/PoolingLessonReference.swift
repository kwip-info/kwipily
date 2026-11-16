import Foundation

/// Indexed scalar windows; no pooling implementation supplies expected values.
public struct PoolingLessonReference {
    public let output:LearningTensor
    public let indices:LearningTensor?
    private let input:LearningTensor,task:PoolingLearningTask
    private let contributors:[[Int]],divisors:[Double],winners:[Int]
    public init(input:LearningTensor,task:PoolingLearningTask)throws {
        guard input.wellFormed,!input.shape.isEmpty,input.shape == [input.shape[0]]+task.sampleShape,(1...2).contains(input.shape[0]) else {throw LearningContractError("Pooling lesson input must match its public channel/spatial shape.")}
        self.input=input;self.task=task
        let size=task.isAverage ? 6:4,extent=task.isAverage ? 4:3,d=task.dimensions,spatial=Int(pow(Double(size),Double(d))),outputSpatial=Int(pow(Double(extent),Double(d)))
        var values:[Double]=[],locations:[Double]=[],contributors:[[Int]]=[],divisors:[Double]=[],winners:[Int]=[]
        func coordinate(_ flat:Int,size:Int)->[Int] {var value=flat;var coords=Array(repeating:0,count:d);for axis in (0..<d).reversed() {coords[axis]=value%size;value/=size};return coords}
        for n in 0..<input.shape[0] {for channel in 0..<2 {for out in 0..<outputSpatial {
            let position=coordinate(out,size:extent);var candidates:[Int]=[]
            for k in 0..<Int(pow(3.0,Double(d))) {
                let offset=coordinate(k,size:3),coords=zip(position,offset).map {$0.0*2-1+$0.1}
                if coords.allSatisfy({(0..<size).contains($0)}) {var cell=0;for c in coords {cell=cell*size+c};candidates.append((n*2+channel)*spatial+cell)}
            }
            guard let first=candidates.first else {throw LearningContractError("Pooling window has no real input position.")}
            contributors.append(candidates)
            if task.isAverage {
                let start=position[0]*2-1,divisor=Double(min(start+3,size+1)-max(start,-1))
                divisors.append(divisor);winners.append(first);values.append(candidates.reduce(0) {$0+input.values[$1]}/divisor)
            } else {
                var winner=first;for candidate in candidates.dropFirst() where input.values[candidate]>input.values[winner] {winner=candidate}
                winners.append(winner);divisors.append(1);values.append(input.values[winner]);locations.append(Double(winner-(n*2+channel)*spatial))
            }
        }}}
        let shape=[input.shape[0],2]+Array(repeating:extent,count:d)
        output = .init(shape:shape,values:values);indices=task.isAverage ? nil:.init(shape:shape,values:locations)
        self.contributors=contributors;self.divisors=divisors;self.winners=winners
    }
    public func inputGradient(for upstream:LearningTensor)throws->LearningTensor {
        guard upstream.wellFormed,upstream.shape == output.shape else {throw LearningContractError("Pooling influence must match its output shape.")}
        var values=Array(repeating:0.0,count:input.values.count)
        for i in upstream.values.indices {
            if task.isAverage {for position in contributors[i] {values[position]+=upstream.values[i]/divisors[i]}}
            else {values[winners[i]]+=upstream.values[i]}
        }
        return .init(shape:input.shape,values:values)
    }
    public static let explanation="Pooling has no learned weights. A max window chooses its largest real input; padded positions behave as negative infinity and cannot replace a negative winner. If several real inputs tie, the first spatial position wins. Its int64 index is a row-major position within one channel’s spatial array; observation and channel coordinates are separate. An average window sums real inputs and divides by its count. Count include pad includes zero positions within the declared padding, but excludes any farther positions in a final partial window. In this task the last average window divides by 2, not 3. Ceiling mode retains that partial window. A gradient records influence from a small change: max routes influence to the chosen winner, while average distributes it by the divisor. Overlapping windows add contributions. Index labels are discrete positions and have no differentiable gradient. Scalar lesson checks use float64 arithmetic; production pooling retains its reviewed float32 values/int64 indices contract."
}
