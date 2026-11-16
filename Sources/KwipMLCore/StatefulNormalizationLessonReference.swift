import Foundation

/// Independent coordinate reduction and remembered-summary oracle; no worker/operator invocation.
public struct StatefulNormalizationLessonReference {
    public let output:LearningTensor
    public let runningMean:[Double],runningVariance:[Double]
    private let task:StatefulNormalizationLearningTask,input:LearningTensor,groups:[[Int]],normalized:[Double],inverse:[Double]
    public init(input:LearningTensor,task:StatefulNormalizationLearningTask,calibrationA:LearningTensor?=nil,calibrationB:LearningTensor?=nil)throws {
        guard input.wellFormed,Array(input.shape.dropFirst())==task.sampleShape else {throw LearningContractError("Use the supplied bounded observation/channel shape.")}
        self.task=task;self.input=input
        let c=input.shape[1],sp=input.shape.dropFirst(2).reduce(1,*),gs=Self.groups(shape:input.shape,componentID:task.componentID)
        var mean=Array(repeating:0.0,count:c),variance=Array(repeating:1.0,count:c)
        if task.remembered {
            for calibration in [calibrationA,calibrationB] {
                guard let t=calibration,t.wellFormed,Array(t.shape.dropFirst())==task.sampleShape else {throw LearningContractError("Remembered-summary objectives require both trusted calibration groups.")}
                var m=Array(repeating:0.0,count:c),v=m
                for (k,ids) in Self.groups(shape:t.shape,componentID:task.componentID).enumerated() {
                    guard ids.count>1 else {throw LearningContractError("Current summaries need more positions or observations.")}
                    let mu=ids.reduce(0.0) {$0+t.values[$1]}/Double(ids.count),vv=ids.reduce(0.0) {$0+pow(t.values[$1]-mu,2)}/Double(ids.count-1),divisor=task.componentID.hasPrefix("instance") ? Double(t.shape[0]):1
                    m[k%c]+=mu/divisor;v[k%c]+=vv/divisor
                }
                for ch in 0..<c {mean[ch]=0.75*mean[ch]+0.25*m[ch];variance[ch]=0.75*variance[ch]+0.25*v[ch]}
            }
        }
        var z=Array(repeating:0.0,count:input.values.count),inv=z,result=z
        let scale=task.scale?.values ?? Array(repeating:1,count:c),bias=task.bias?.values ?? Array(repeating:0,count:c)
        for ids in gs {
            guard ids.count>1 else {throw LearningContractError("A comparison group needs more than one value.")}
            let mu=ids.reduce(0.0) {$0+input.values[$1]}/Double(ids.count),v=ids.reduce(0.0) {$0+pow(input.values[$1]-mu,2)}/Double(ids.count)
            for i in ids {let ch=(i/sp)%c;inv[i]=1/sqrt((task.remembered ? variance[ch]:v)+0.125);z[i]=(input.values[i]-(task.remembered ? mean[ch]:mu))*inv[i];result[i]=z[i]*scale[ch]+bias[ch]}
        }
        guard (result+mean+variance).allSatisfy(\.isFinite) else {throw LearningContractError("Summary arithmetic exceeded finite range.")}
        self.output = .init(shape:input.shape,values:result);self.runningMean=mean;self.runningVariance=variance;self.groups=gs;self.normalized=z;self.inverse=inv
    }
    private static func groups(shape:[Int],componentID:String)->[[Int]] {
        let n=shape[0],c=shape[1],s=shape.dropFirst(2).reduce(1,*)
        if componentID.hasPrefix("batch") {return (0..<c).map {ch in (0..<n).flatMap {a in (0..<s).map {(a*c+ch)*s+$0}}}}
        if componentID.hasPrefix("instance") {return (0..<n).flatMap {a in (0..<c).map {ch in (0..<s).map {(a*c+ch)*s+$0}}}}
        return (0..<n).flatMap {a in (0..<2).map {g in (g*c/2..<(g+1)*c/2).flatMap {ch in (0..<s).map {(a*c+ch)*s+$0}}}}
    }
    public func expected(for upstream:LearningTensor)throws->[String:LearningTensor] {
        guard upstream.wellFormed,upstream.shape==input.shape else {throw LearningContractError("Output influence must match the supplied array.")}
        let c=input.shape[1],s=input.shape.dropFirst(2).reduce(1,*),scale=task.scale?.values ?? Array(repeating:1,count:c)
        var dx=Array(repeating:0.0,count:input.values.count),dg=Array(repeating:0.0,count:c),db=dg
        for ids in groups {
            let h=ids.map {upstream.values[$0]*scale[($0/s)%c]},mh=h.reduce(0,+)/Double(ids.count),mhz=zip(ids,h).reduce(0.0) {$0+$1.1*normalized[$1.0]}/Double(ids.count)
            for (k,i) in ids.enumerated() {let ch=(i/s)%c;dx[i]=inverse[i]*(task.remembered ? h[k]:h[k]-mh-normalized[i]*mhz);dg[ch]+=upstream.values[i]*normalized[i];db[ch]+=upstream.values[i]}
        }
        var tensors=["output":output,"evaluationOutput":output,"inputGradient":LearningTensor(shape:input.shape,values:dx)]
        if task.affine {tensors["scaleGradient"] = .init(shape:[c],values:dg);tensors["biasGradient"] = .init(shape:[c],values:db)}
        if task.remembered {for prefix in ["","postEval"] {tensors[prefix.isEmpty ? "runningMean":prefix+"RunningMean"] = .init(shape:[c],values:runningMean);tensors[prefix.isEmpty ? "runningVariance":prefix+"RunningVariance"] = .init(shape:[c],values:runningVariance);tensors[prefix.isEmpty ? "batchesTracked":prefix+"BatchesTracked"] = .init(shape:[1],values:[task.componentID.hasPrefix("batch") ? 2:0])}}
        guard tensors.values.allSatisfy(\.wellFormed) else {throw LearningContractError("Measured-reference influence exceeded finite bounds.")};return tensors
    }
    public static let explanation="Batch normalization compares a channel across observations and positions. Instance normalization compares positions within one observation/channel. Group normalization compares equal teams of channels within one observation. Current spread uses population variance; remembered spread uses corrected variance. Remembered summaries update with a declared share, and evaluation keeps them fixed. Channel scale/offset contributions accumulate across each channel's uses."
}
