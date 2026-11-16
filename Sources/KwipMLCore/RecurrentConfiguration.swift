import Foundation

/// Explicit dense sequence/state geometry; hidden state never persists inside a component.
public struct RecurrentConfiguration: Equatable, Sendable {
    public static let componentIDs:Set<String> = ["rnn","lstm","gru","rnn_cell","lstm_cell","gru_cell"]
    public struct Parameter:Equatable,Sendable {public let name:String;public let shape:[Int]}
    public struct Resolved:Equatable,Sendable {
        public let outputs:[String:BatchedTensorLayout]
        public let outputElementsPerObservation:Int
        public let gateElementsPerObservation:Int
        public let workPerObservation:Int
    }
    public let componentID:String
    public let values:[String:JSONValue]
    public var isCell:Bool {componentID.hasSuffix("_cell")}
    public var isLSTM:Bool {componentID.hasPrefix("lstm")}
    public var inputSize:Int {values["input_size"]!.integerValue!}
    public var hiddenSize:Int {values["hidden_size"]!.integerValue!}
    public var layerCount:Int {values["num_layers"]?.integerValue ?? 1}
    public var directionCount:Int {values["bidirectional"] == .bool(true) ? 2:1}
    public var projectionSize:Int {values["proj_size"]?.integerValue ?? 0}
    public var stateWidth:Int {projectionSize == 0 ? hiddenSize:projectionSize}
    public var gates:Int {isLSTM ? 4:componentID.hasPrefix("gru") ? 3:1}
    public var configuredInputs:[String] {values["state_mode"] == .string("zero") ? ["in"]:isLSTM ? ["in","hidden","cell"]:["in","hidden"]}
    public var outputPorts:[String] {isLSTM ? ["out","hidden","cell"]:["out","hidden"]}
    public var stochastic:Bool {!isCell && layerCount>1 && (values["dropout"]?.doubleValue ?? 0)>0}
    public init(componentID:String,values supplied:[String:JSONValue]) throws {
        guard Self.componentIDs.contains(componentID) else {throw GraphConfigurationError("Choose a supported sequence or single-step memory component.")}
        var c:[String:JSONValue] = ["input_size":2,"hidden_size":2,"bias":true,"state_mode":.string("zero")]
        if !componentID.hasSuffix("_cell") {c.merge(["num_layers":1,"bidirectional":false,"dropout":0]) {_,new in new}}
        if componentID.hasPrefix("rnn") {c["nonlinearity"] = .string("tanh")}
        if componentID == "lstm" {c["proj_size"]=0}
        guard Set(supplied.keys).isSubset(of:Set(c.keys)) else {throw GraphConfigurationError("Use only the declared recurrent controls. Connect state explicitly; weights are learned parameters.")}
        c.merge(supplied) {_,new in new}
        for (key,bounds) in [("input_size",1...4096),("hidden_size",1...512),("num_layers",1...8),("proj_size",0...511)] where c[key] != nil {
            guard let v=c[key]?.doubleValue,v.isFinite,v.rounded(.towardZero)==v,v>=Double(bounds.lowerBound),v<=Double(bounds.upperBound) else {throw GraphConfigurationError("\(key) must be a whole number from \(bounds.lowerBound) to \(bounds.upperBound).")}
            c[key] = .integer(Int64(v))
        }
        for key in ["bias","bidirectional"] where c[key] != nil {guard c[key]?.boolValue != nil else {throw GraphConfigurationError("\(key) must be a Boolean switch.")}}
        guard let state=c["state_mode"]?.stringValue,["zero","supplied"].contains(state) else {throw GraphConfigurationError("Choose zero-reset or supplied state.")}
        if let v=c["nonlinearity"] {guard let name=v.stringValue,["tanh","relu"].contains(name) else {throw GraphConfigurationError("Choose tanh or relu for this memory component.")}}
        if c["dropout"] != nil {
            guard let p=c["dropout"]?.doubleValue,p.isFinite,(0...1).contains(p) else {throw GraphConfigurationError("Inter-layer dropout must be a finite probability from 0 to 1.")}
            guard p == 0 || c["num_layers"]!.integerValue!>1 else {throw GraphConfigurationError("Inter-layer dropout needs two or more layers. Choose two layers or dropout zero.")}
            c["dropout"] = .number(p)
        }
        guard (c["proj_size"]?.integerValue ?? 0)<c["hidden_size"]!.integerValue! else {throw GraphConfigurationError("Projection width must be below hidden width; zero disables projection.")}
        self.componentID=componentID;values=c
    }
    public var parameterShapes:[Parameter] {
        var result:[Parameter]=[]
        for layer in 0..<layerCount {
            let width=layer == 0 ? inputSize:directionCount*stateWidth
            for direction in 0..<directionCount {
                let suffix=isCell ? "":"_l\(layer)"+(direction == 1 ? "_reverse":"")
                result.append(.init(name:"weight_ih"+suffix,shape:[gates*hiddenSize,width]))
                result.append(.init(name:"weight_hh"+suffix,shape:[gates*hiddenSize,stateWidth]))
                if values["bias"] == .bool(true) {
                    result.append(.init(name:"bias_ih"+suffix,shape:[gates*hiddenSize]))
                    result.append(.init(name:"bias_hh"+suffix,shape:[gates*hiddenSize]))
                }
                if projectionSize>0 {result.append(.init(name:"weight_hr"+suffix,shape:[stateWidth,hiddenSize]))}
            }
        }
        return result
    }
    public var parameterCount:Int {parameterShapes.reduce(0) {$0+$1.shape.reduce(1,*)}}
    public func supports(device:String)->Bool {["cpu","mps"].contains(device) && !(device == "mps" && projectionSize>0)}
    public func resolved(layouts:[BatchedTensorLayout])throws->Resolved {
        guard layouts.count == configuredInputs.count else {throw GraphConfigurationError("Connect exactly the enabled input and supplied-state ports.")}
        let sample=layouts[0].sample.dimensions
        guard sample.count == (isCell ? 1:2),sample.last == inputSize,isCell || (1...4096).contains(sample[0]) else {throw GraphConfigurationError("Connect the exact feature width and one to 4096 ordered time steps.")}
        let t=isCell ? 1:sample[0],hidden=isCell ? [hiddenSize]:[layerCount*directionCount,stateWidth],cell=isCell ? [hiddenSize]:[layerCount*directionCount,hiddenSize]
        if layouts.count>1 {guard layouts[1].sample.dimensions == hidden else {throw GraphConfigurationError("Supplied hidden state must match layers, directions and hidden or projected width; observations stay first.")}}
        if layouts.count>2 {guard layouts[2].sample.dimensions == cell else {throw GraphConfigurationError("Supplied cell memory must match layers, directions and cell width; observations stay first.")}}
        var outputs:[String:BatchedTensorLayout] = ["out":try .init(sample:isCell ? [hiddenSize]:[t,directionCount*stateWidth]),"hidden":try .init(sample:hidden)]
        if isLSTM {outputs["cell"]=try .init(sample:cell)}
        let physical=outputs.reduce(0) {$0+(isCell && $1.key == "hidden" ? 0:$1.value.sample.elementCount)}
        let gate=t*layerCount*directionCount*gates*hiddenSize
        let work=t*(0..<layerCount).reduce(0) {$0+directionCount*(gates*hiddenSize*(($1 == 0 ? inputSize:directionCount*stateWidth)+stateWidth)+(projectionSize>0 ? stateWidth*hiddenSize:0))}
        guard parameterCount<=2_000_000,[physical,gate,work].allSatisfy({$0<=100_000_000}),layouts.allSatisfy({$0.sample.elementCount<=100_000_000}) else {throw GraphConfigurationError("Memory component exceeds the learned-parameter, output, gate storage or computation budget. Reduce width, layers or time steps.")}
        return .init(outputs:outputs,outputElementsPerObservation:physical,gateElementsPerObservation:gate,workPerObservation:work)
    }
}
