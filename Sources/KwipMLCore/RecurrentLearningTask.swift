import Foundation

public enum RecurrentLearningTask:String,CaseIterable,Codable {
    case rnn="rnn-sequence-memory",lstm="lstm-separate-memory",gru="gru-reset-memory",rnnCell="rnn-cell-step",lstmCell="lstm-cell-step",gruCell="gru-cell-step"
    public var componentID:String {switch self {case .rnn:return "rnn";case .lstm:return "lstm";case .gru:return "gru";case .rnnCell:return "rnn_cell";case .lstmCell:return "lstm_cell";case .gruCell:return "gru_cell"}}
    public var isCell:Bool {componentID.hasSuffix("_cell")}
    public var isLSTM:Bool {componentID.hasPrefix("lstm")}
    public var isGRU:Bool {componentID.hasPrefix("gru")}
    public var componentTitle:String {switch self {case .rnn:return "RNN";case .lstm:return "LSTM";case .gru:return "GRU";case .rnnCell:return "RNNCell";case .lstmCell:return "LSTMCell";case .gruCell:return "GRUCell"}}
    public var title:String {componentTitle+" · "+(isCell ? "Continue one observation step":isLSTM ? "Carry hidden and cell memory":isGRU ? "Control which earlier information is reused":"Remember a sequence of measurements")}
    public var sourceRoles:[String] {isLSTM ? ["input","hidden","cell"]:["input","hidden"]}
    public var outputRoles:[String] {isLSTM ? ["result","hidden","cell"]:["result","hidden"]}
    public var sampleShapes:[String:[Int]] {var result=["input":isCell ? [2]:[3,2],"hidden":isCell ? [2]:[1,2]];if isLSTM {result["cell"]=isCell ? [2]:[1,2]};return result}
    public var outputSampleShapes:[String:[Int]] {var result=["result":sampleShapes["input"]!,"hidden":sampleShapes["hidden"]!];if isLSTM {result["cell"]=sampleShapes["cell"]!};return result}
    public var configuration:[String:JSONValue] {var c:[String:JSONValue]=["input_size":2,"hidden_size":2,"bias":true,"state_mode":.string("supplied")];if !isCell {c["num_layers"]=1;c["bidirectional"]=false;c["dropout"] = .number(0)};if componentID.hasPrefix("rnn") {c["nonlinearity"] = .string("relu")};if self == .lstm {c["proj_size"]=0};return c}
    public var parameterShapes:[String:[Int]] {let g=isLSTM ? 4:isGRU ? 3:1,s=isCell ? "":"_l0";return ["weight_ih"+s:[g*2,2],"weight_hh"+s:[g*2,2],"bias_ih"+s:[g*2],"bias_hh"+s:[g*2]]}
    public var controlInstructions:String {"Set Input width and Hidden width to 2, Starting state to supplied and Use bias on. "+(!isCell ? "Set Layer count to 1, Reverse direction too off and Between-layer dropout to 0. ":"")+(componentID.hasPrefix("rnn") ? "Choose relu for Behavior; it keeps positive values and replaces negative values with zero. ":"")+(self == .lstm ? "Set Projected hidden width to 0. ":"")+"Keep each result and state connected to its matching Output role."}
    public var explanation:String {"A sequence is measurements in time order. State is information carried from the previous step. Hidden state is the summary used to produce the next result. "+(isLSTM ? "Cell state is a separate longer-lived memory. A gate is a number between zero and one choosing how much to keep, add or reveal. ":isGRU ? "A gate is a number between zero and one that chooses how much information passes. GRU uses reset and update gates to blend a new candidate with earlier hidden state. ":"The same small learned table combines the current measurement and earlier hidden state at each step. ")+(isCell ? "Result and hidden show the same newly produced hidden state for this single step. ":"Result shows every time position; hidden shows the summary after the final position. ")+"Each observation has its own state. These blocks never silently retain state between runs. A gradient, shown as influence, measures how a small change affects the results; the checker measures it for you."}
    public var instructions:String {explanation+" Build: add one Input for each supplied role ("+sourceRoles.joined(separator:", ")+"). Connect input to in and the state sources to their matching ports. Add separate Model Output blocks for "+outputRoles.joined(separator:", ")+"; result uses out, while state uses its named port. Configure: "+controlInstructions+" Check: choose Check construction and compare the time-ordered result and final-state tables. The checker supplies varied learned tables, input values and starting states and checks every influence. No code or handwritten derivatives are required."}
}
public enum RecurrentAssessment {
    public static func plan(task:RecurrentLearningTask,runtimeVersion:String,seed:UInt64)throws->LearningAssessmentPlan {
        var state=seed
        func tensor(_ shape:[Int],positive:Bool=false)->LearningTensor {.init(shape:shape,values:(0..<shape.reduce(1,*)).map {_ in state=state &* 6364136223846793005 &+ 1442695040888963407;let v=Double(Int((state>>32)%19)-9)/20;return positive ? abs(v)+0.05:v})}
        let fixtures=try (0..<4).map {index->LearningFixture in
            let n=index%3+1;var inputs:[String:LearningTensor]=[:]
            for role in task.sourceRoles {inputs[role]=tensor([n]+task.sampleShapes[role]!,positive:true)}
            for name in task.parameterShapes.keys.sorted() {inputs["parameter."+name]=tensor(task.parameterShapes[name]!,positive:task.componentID.hasPrefix("rnn"))}
            for role in task.outputRoles {inputs["upstream."+role]=tensor([n]+task.outputSampleShapes[role]!)}
            return .init(id:"\(task.rawValue)-\(seed)-\(index)",inputs:inputs,expected:try RecurrentLessonReference(inputs:inputs,task:task).expected)
        }
        let criteria=fixtures[0].expected.keys.sorted().map {key in LearningCriterion(id:key,description:key.hasSuffix("Gradient") ? "Trace the influence of the supplied measurements, state and learned tables.":"Produce the requested time-ordered result or final state for each observation.",nextStep:task.controlInstructions,rule:.tensor(key:key,absoluteTolerance:2e-6,relativeTolerance:2e-5))}
        let plan=LearningAssessmentPlan(id:"assessment."+task.rawValue,objectiveID:"recurrent."+task.rawValue,task:task.instructions,runtime:.init(backend:"pytorch",version:runtimeVersion,capabilities:[task.componentID:ContractVersion()]),fixtureSeed:seed,fixtures:fixtures,criteria:criteria);try plan.validate();return plan
    }
}
