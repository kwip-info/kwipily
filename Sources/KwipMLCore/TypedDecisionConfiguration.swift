import Foundation
import CryptoKit

/// An inert immutable question resource, shared by value/revision across Train and Decide.
/// Scientific hashes include the actual labels, order and instructions; History keeps originals.
public enum TypedDecisionConfiguration {
    public static let trainingDefaults:[String:JSONValue] = try! JSONDecoder().decode([String:JSONValue].self,from:Data(#"{"task":"typedDecisions","questionSet":{"schemaVersion":1,"id":"ticket-triage","revision":1,"name":"Ticket triage","questions":[{"id":"category","type":"choice","instructions":"Which team should handle this ticket?","options":[{"id":"billing","label":"Billing and payments"},{"id":"technical","label":"Technical problems"},{"id":"account","label":"Account access"}]}]},"stateColumns":["text"],"targetColumns":{"category":"label"},"targetModes":{},"groupColumn":"case_id","mode":"head","objective":"softCE","encoderLearningRate":2.5e-05,"headLearningRate":0.0001,"accumulation":1,"calibrationFraction":0.0,"targetError":0.1,"activationCheckpointing":false,"startMode":"base","resumeRecord":"","resumeEpoch":0,"maximumTokens":512}"#.utf8))
    public static let operatorID="kwip.data.decide"
    public static let defaultQuestionSet:[String:JSONValue] = try! JSONDecoder().decode([String:JSONValue].self,from:Data(#"{"schemaVersion":1,"id":"ticket-triage","revision":1,"name":"Ticket triage","questions":[{"id":"category","type":"choice","instructions":"Which team should handle this ticket?","options":[{"id":"billing","label":"Billing and payments"},{"id":"technical","label":"Technical problems"},{"id":"account","label":"Account access"}]}]}"#.utf8))
    public static let defaults:[String:JSONValue] = ["stateColumns":.array([.string("text")]),"identityColumn":.string(""),"questionSet":.object(defaultQuestionSet),"questionsFromModel":false,"batchRows":1,"maximumRows":1000,"maximumScoreRows":100000]
    public static func validKey(_ text:String)->Bool {text.range(of:#"^[A-Za-z][A-Za-z0-9_-]{0,79}$"#,options:.regularExpression) != nil}
    public static func validateQuestionSet(_ value:[String:JSONValue])throws {
        guard Set(value.keys)==["schemaVersion","id","revision","name","questions"],value["schemaVersion"] == .integer(1),let id=value["id"]?.stringValue,validKey(id),let revision=value["revision"]?.integerValue,(1...1_000_000).contains(revision),let name=value["name"]?.stringValue,!name.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,name.count<=160,case .array(let questions)=value["questions"],(1...16).contains(questions.count) else {throw DataFlowContractError("Choose a complete named question-set revision with1–16 questions.")}
        var ids=Set<String>()
        for entry in questions {
            guard case .object(let q)=entry,Set(q.keys)==["id","type","instructions","options"],let id=q["id"]?.stringValue,validKey(id),ids.insert(id).inserted,let type=q["type"]?.stringValue,["choice","score","noul"].contains(type),let instructions=q["instructions"]?.stringValue,!instructions.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,instructions.count<=2000,case .array(let options)=q["options"],(2...32).contains(options.count) else{throw DataFlowContractError("Each question needs a unique key, type, instructions and2–32 options.")}
            var keys=Set<String>(),labels=Set<String>(),ordered:[String]=[]
            for option in options {
                guard case .object(let o)=option,Set(o.keys)==["id","label"],let key=o["id"]?.stringValue,validKey(key),keys.insert(key).inserted,let label=o["label"]?.stringValue,!label.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,label.count<=240,labels.insert(label).inserted else{throw DataFlowContractError("Options need unique stable keys and labels.")};ordered.append(key)
            }
            if type=="noul",ordered != ["false","true"] {throw DataFlowContractError("Yes/no options must remain false then true.")}
        }
    }
    public static func questionHash(_ value:[String:JSONValue])throws->String {try validateQuestionSet(value);let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes];return SHA256.hash(data:try encoder.encode(value)).map{String(format:"%02x",$0)}.joined()}
    public static func validate(_ c:[String:JSONValue])throws {
        guard Set(c.keys)==Set(defaults.keys),case .array(let columns)=c["stateColumns"],(1...16).contains(columns.count),columns.allSatisfy({if let n=$0.stringValue{return !n.isEmpty && n.count<=160};return false}),Set(columns.compactMap(\.stringValue)).count==columns.count,let identity=c["identityColumn"]?.stringValue,identity.count<=160,case .bool=c["questionsFromModel"],case .object(let questions)=c["questionSet"] else {throw DataFlowContractError("Choose unique state columns, source identity and a question set.")}
        for (key,bound) in [("batchRows",8),("maximumRows",10000),("maximumScoreRows",200000)] {guard let n=c[key]?.integerValue,(1...bound).contains(n)else{throw DataFlowContractError("Decide row/score/CPU batch limit exceeds admission.")}}
        try validateQuestionSet(questions)
    }
    public static func validateTraining(_ supplied:[String:JSONValue])throws->[String:JSONValue] {
        guard Set(supplied.keys).isSubset(of:Set(trainingDefaults.keys)) else{throw DataFlowContractError("Unknown typed-decision training control.")}
        let c=trainingDefaults.merging(supplied){_,new in new}
        guard c["task"] == .string("typedDecisions"),["head","full"].contains(c["mode"]?.stringValue ?? ""),c["objective"] == .string("softCE"),["base","weights"].contains(c["startMode"]?.stringValue ?? ""),c["resumeEpoch"] == .integer(0),case .object(let q)=c["questionSet"],case .array(let states)=c["stateColumns"],!states.isEmpty,states.count<=16,Set(states.compactMap(\.stringValue)).count==states.count,states.allSatisfy({($0.stringValue?.isEmpty)==false}),case .object(let targets)=c["targetColumns"],case .object(let modes)=c["targetModes"],let group=c["groupColumn"]?.stringValue,!group.isEmpty,group.count<=160,case .bool=c["activationCheckpointing"] else{throw DataFlowContractError("Choose CPU typed-decision head/full training, state/target mappings and source groups.")}
        try validateQuestionSet(q)
        guard case .array(let questions)=q["questions"] else{throw DataFlowContractError("Missing questions.")}
        let ids=Set(questions.compactMap{if case .object(let q)=$0{return q["id"]?.stringValue};return nil})
        guard Set(targets.keys)==ids,targets.values.allSatisfy({if let n=$0.stringValue{return !n.isEmpty && n.count<=160};return false}),Set(modes.keys).isSubset(of:ids),modes.values.allSatisfy({[JSONValue.string("label"),.string("distribution")].contains($0)})else{throw DataFlowContractError("Each question needs a target column and explicit label/distribution mode.")}
        for (key,lo,hi) in [("headLearningRate",1e-8,1.0),("encoderLearningRate",1e-8,1.0),("calibrationFraction",0.0,0.5),("targetError",0.0,0.5)] {guard let n=c[key]?.doubleValue,n.isFinite,(lo...hi).contains(n) else{throw DataFlowContractError("Invalid typed-decision learning rate or calibration allocation.")}}
        guard let tokens=c["maximumTokens"]?.integerValue,(1...1024).contains(tokens),let a=c["accumulation"]?.integerValue,(1...1024).contains(a) else{throw DataFlowContractError("Choose1–1024 accumulation steps.")}
        return c
    }
    public static var trainingSchema:JSONValue {
        let props=trainingDefaults.mapValues {v->JSONValue in
            let type:String
            switch v {case .string:type="string";case .integer:type="integer";case .number:type="number";case .bool:type="boolean";case .array:type="array";default:type="object"}
            return .object(["type":.string(type),"default":v])
        }
        var properties=props;properties["task"] = .object(["const":.string("typedDecisions")])
        return .object(["type":.string("object"),"required":.array([.string("task")]),"additionalProperties":false,"properties":.object(properties)])
    }

}

extension WorkflowNodePayload {
    var isDecisionModelProvider:Bool {switch self {case .frozenModel:return true;case .train(let config):return config["fineTuning"] != nil;default:return false}}
}
