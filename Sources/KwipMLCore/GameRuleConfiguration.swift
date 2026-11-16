import Foundation

/// Pure configuration envelope. Semantic rule validation belongs to the isolated evaluator.
public enum GameRuleConfiguration {
    public static let operatorID="kwip.data.gameRules"
    public static let lessonID="game-rules"
    public static let source=WorkflowNodeAddress(domain:.data,localID:"rules")
    public static let examples=["take-away","placement","gravity"]
    public static func example(_ name:String)throws->[String:JSONValue] {
        guard examples.contains(name),let url=Bundle.module.url(forResource:name,withExtension:"json",subdirectory:"GameRules") ?? Bundle.module.url(forResource:name,withExtension:"json") else {throw DataFlowContractError("Bundled game definition is unavailable.")}
        return try decode(Data(contentsOf:url))
    }
    public static func decode(_ bytes:Data)throws->[String:JSONValue] {
        guard bytes.count<=131072 else {throw DataFlowContractError("Game definition exceeds 128 KiB.")}
        guard case .object(let value)=try decodeValue(bytes) else {throw DataFlowContractError("Use a JSON definition object.")}
        let required:Set<String>=["version","name","state","actions","currentPlayer","legal","transitions","terminal","observations","reward","invariants","tests","parameters"]
        guard Set(value.keys)==required,value["version"] == .integer(1),let name=value["name"]?.stringValue,!name.isEmpty,name.count<=120,
              case .object?=value["state"],case .object?=value["actions"],case .array?=value["tests"] else {throw DataFlowContractError("Use a version1 definition with typed state, actions, rules, observations and tests. Validate rules before applying.")}
        return value
    }
    public static func decodeValue(_ bytes:Data)throws->JSONValue {
        guard bytes.count<=131072 else {throw DataFlowContractError("Game JSON exceeds 128 KiB.")}
        // JSONDecoder otherwise silently accepts duplicate keys. Scan string keys before decoding.
        let b=Array(bytes);var i=0;var stack:[Set<String>?]=[]
        while i<b.count {
            switch b[i] {
            case 123:stack.append(Set<String>());i+=1
            case 91:stack.append(nil);i+=1
            case 125,93:if !stack.isEmpty {stack.removeLast()};i+=1
            case 34:
                let start=i;i+=1
                while i<b.count {if b[i]==92 {i+=2;continue};if b[i]==34 {i+=1;break};i+=1}
                var next=i;while next<b.count && [9,10,13,32].contains(b[next]) {next+=1}
                if next<b.count,b[next]==58,!stack.isEmpty,var keys=stack[stack.count-1] {
                    let key=try JSONDecoder().decode(String.self,from:Data(b[start..<min(i,b.count)]))
                    guard keys.insert(key).inserted else {throw DataFlowContractError("Duplicate JSON key: \(key)")};stack[stack.count-1]=keys
                }
            default:i+=1
            }
        }
        return try JSONDecoder().decode(JSONValue.self,from:bytes)
    }
    public static func validate(_ values:[String:JSONValue])throws {
        guard Set(values.keys)==["definition"],let definition=values["definition"] else {throw DataFlowContractError("Game definition requires its canonical definition object.")}
        _=try decode(JSONEncoder().encode(definition))
    }
    public static func document()throws->WorkflowDocument {
        let definition=try example("take-away"),tests=WorkflowNodeAddress(domain:.data,localID:"tests"),trace=WorkflowNodeAddress(domain:.data,localID:"trace")
        return .init(projectID:UUID().uuidString,name:"Author and test game rules",nodes:[
            .init(address:source,name:"Author game rules",x:60,y:120,payload:.dataOperator(operatorID:operatorID,config:["definition":.object(definition)])),
            .init(address:tests,name:"Inspect rule tests",x:460,y:70,payload:.dataOperator(operatorID:"kwip.data.view",config:[:])),
            .init(address:trace,name:"Inspect move trace",x:460,y:330,payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))],edges:[
                .init(id:"tests",from:.init(address:source,port:"tests"),to:.init(address:tests,port:"table"),kind:.table),
                .init(id:"trace",from:.init(address:source,port:"trace"),to:.init(address:trace,port:"table"),kind:.table)])
    }
}
