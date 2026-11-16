import Foundation

/// Scalar gate arithmetic and coordinate finite differences. No adapter/native module is a gold source.
public struct RecurrentLessonReference {
    public let expected:[String:LearningTensor]
    public init(inputs:[String:LearningTensor],task:RecurrentLearningTask)throws {
        let keys=Set(task.sourceRoles+task.parameterShapes.keys.map {"parameter."+$0}+task.outputRoles.map {"upstream."+$0})
        guard Set(inputs.keys)==keys,let n=inputs["input"]?.shape.first,(1...3).contains(n),inputs.values.allSatisfy(\.wellFormed) else {throw LearningContractError("Use the bounded supplied measurement, state, learned tables and result influence.")}
        for role in task.sourceRoles {guard inputs[role]!.shape == [n]+task.sampleShapes[role]! else {throw LearningContractError("A supplied state or measurement shape differs.")}}
        for (name,shape) in task.parameterShapes {guard inputs["parameter."+name]!.shape==shape else {throw LearningContractError("A supplied learned table shape differs.")}}
        for role in task.outputRoles {guard inputs["upstream."+role]!.shape == [n]+task.outputSampleShapes[role]! else {throw LearningContractError("Result influence must match its named result.")}}
        var answer=Self.forward(inputs,task:task),scratch=inputs
        func weighted()->Double {let output=Self.forward(scratch,task:task);return task.outputRoles.reduce(0) {total,role in total+zip(output[role]!.values,scratch["upstream."+role]!.values).reduce(0) {$0+$1.0*$1.1}}}
        for key in (task.sourceRoles+task.parameterShapes.keys.sorted().map {"parameter."+$0}) {
            var gradients:[Double]=[]
            for i in inputs[key]!.values.indices {let v=inputs[key]!.values[i];scratch[key]!.values[i]=v+1e-6;let plus=weighted();scratch[key]!.values[i]=v-1e-6;let minus=weighted();scratch[key]!.values[i]=v;gradients.append((plus-minus)/2e-6)}
            answer[key+"Gradient"] = .init(shape:inputs[key]!.shape,values:gradients)
        }
        expected=answer
    }
    private static func forward(_ f:[String:LearningTensor],task:RecurrentLearningTask)->[String:LearningTensor] {
        let n=f["input"]!.shape[0],steps=task.isCell ? 1:3,g=task.isLSTM ? 4:task.isGRU ? 3:1,s=task.isCell ? "":"_l0"
        let wi=f["parameter.weight_ih"+s]!.values,wh=f["parameter.weight_hh"+s]!.values,bi=f["parameter.bias_ih"+s]!.values,bh=f["parameter.bias_hh"+s]!.values
        func sigmoid(_ v:Double)->Double {1/(1+exp(-v))}
        var result:[Double]=[],hidden:[Double]=[],cells:[Double]=[]
        for row in 0..<n {
            var h=Array(f["hidden"]!.values[row*2..<row*2+2]),c=task.isLSTM ? Array(f["cell"]!.values[row*2..<row*2+2]):[]
            for t in 0..<steps {
                let x=Array(f["input"]!.values[(row*steps+t)*2..<(row*steps+t+1)*2]);var a:[Double]=[],b:[Double]=[]
                for j in 0..<g*2 {a.append(wi[j*2]*x[0]+wi[j*2+1]*x[1]+bi[j]);b.append(wh[j*2]*h[0]+wh[j*2+1]*h[1]+bh[j])}
                if task.isLSTM {c=(0..<2).map {j in sigmoid(a[2+j]+b[2+j])*c[j]+sigmoid(a[j]+b[j])*tanh(a[4+j]+b[4+j])};h=(0..<2).map {j in sigmoid(a[6+j]+b[6+j])*tanh(c[j])}}
                else if task.isGRU {let old=h;h=(0..<2).map {j in let reset=sigmoid(a[j]+b[j]),update=sigmoid(a[2+j]+b[2+j]);return (1-update)*tanh(a[4+j]+reset*b[4+j])+update*old[j]}}
                else {h=(0..<2).map {max(0,a[$0]+b[$0])}}
                result+=h
            }
            hidden+=h;cells+=c
        }
        var output=["result":LearningTensor(shape:[n]+task.outputSampleShapes["result"]!,values:result),"hidden":LearningTensor(shape:[n]+task.outputSampleShapes["hidden"]!,values:hidden)]
        if task.isLSTM {output["cell"] = .init(shape:[n]+task.outputSampleShapes["cell"]!,values:cells)}
        return output
    }
}
