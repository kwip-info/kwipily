import Foundation

/// Independent scalar bin/window enumeration; no backend pooling supplies expected values.
public struct AdaptivePoolingLessonReference {
    public let output:LearningTensor
    public let indices:LearningTensor?
    private let input:LearningTensor,task:AdaptivePoolingLearningTask
    private let contributors:[[Int]],divisors:[Double],winners:[Int]
    public init(input:LearningTensor,task:AdaptivePoolingLearningTask)throws {
        guard input.wellFormed,!input.shape.isEmpty,input.shape == [input.shape[0]]+task.sampleShape,(1...2).contains(input.shape[0]) else {throw LearningContractError("Pooling lesson input must match its public channel and spatial shape.")}
        self.input=input;self.task=task
        let spatial=Array(task.sampleShape.dropFirst()),outSpatial=Array(task.outputSampleShape.dropFirst())
        let spatialCount=spatial.reduce(1,*),outputCount=outSpatial.reduce(1,*)
        var values:[Double]=[],locations:[Double]=[],contributors:[[Int]]=[],divisors:[Double]=[],winners:[Int]=[]
        func coordinate(_ flat:Int,shape:[Int])->[Int] {var value=flat;var coords=Array(repeating:0,count:shape.count);for axis in shape.indices.reversed() {coords[axis]=value%shape[axis];value/=shape[axis]};return coords}
        for n in 0..<input.shape[0] {for channel in 0..<2 {for cell in 0..<outputCount {
            let point=coordinate(cell,shape:outSpatial)
            var starts:[Int]=[],ends:[Int]=[]
            for axis in spatial.indices {
                if task == .averageTwo {let stride=axis == 0 ? 2:3;starts.append(point[axis]*stride-1);ends.append(starts[axis]+3)}
                else {starts.append(point[axis]*spatial[axis]/outSpatial[axis]);ends.append(((point[axis]+1)*spatial[axis]+outSpatial[axis]-1)/outSpatial[axis])}
            }
            let windowShape=zip(starts,ends).map {$0.1-$0.0};var candidates:[Int]=[]
            for offset in 0..<windowShape.reduce(1,*) {
                let local=coordinate(offset,shape:windowShape),source=zip(local,starts).map {$0.0+$0.1}
                if zip(source,spatial).allSatisfy({(0..<$0.1).contains($0.0)}) {var flat=0;for (value,length) in zip(source,spatial) {flat=flat*length+value};candidates.append((n*2+channel)*spatialCount+flat)}
            }
            guard let first=candidates.first else {throw LearningContractError("Pooling group has no real input position.")}
            contributors.append(candidates)
            if task.isAverage {
                let divisor=task == .averageTwo ? Double(spatial.indices.reduce(1) {$0*(min(ends[$1],spatial[$1]+1)-starts[$1])}):Double(candidates.count)
                divisors.append(divisor);winners.append(first);values.append(candidates.reduce(0) {$0+input.values[$1]}/divisor)
            } else {
                var winner=first;for candidate in candidates.dropFirst() where input.values[candidate]>input.values[winner] {winner=candidate}
                divisors.append(1);winners.append(winner);values.append(input.values[winner]);locations.append(Double(winner-(n*2+channel)*spatialCount))
            }
        }}}
        let shape=[input.shape[0]]+task.outputSampleShape
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
    public static let explanation="A pooling block has no learned weights. An average combines real values and divides their total by a count; a maximum selects the first largest real value. Adaptive pooling chooses input groups to meet the requested output size. Group boundaries round the start down and the end up, so neighboring groups can overlap when input and output sizes do not divide evenly. A null map size keeps that input direction. Larger outputs reuse input positions rather than inventing new measurements. Ordinary average windows count declared zero padding when requested, but extra cells beyond declared padding in a partial final window never affect the count. Divisor override replaces the count with an explicit nonzero signed integer; a negative divisor reverses the output and influence signs. A gradient measures input influence: averages share it across contributing positions, maxima route it to their first winner, and repeated uses add contributions. Maximum indices are zero-based reading-order positions within a channel and have no differentiable gradient. Lesson checks use float64 arithmetic; production graph values remain float32 with optional int64 indices."
}
