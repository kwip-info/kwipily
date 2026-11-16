import Foundation

public struct SequenceWindowColumn:Equatable {
    public let name:String
    public let dtype:String
    public let nullable:Bool
    public init(name:String,dtype:String,nullable:Bool=false){self.name=name;self.dtype=dtype;self.nullable=nullable}
}

public enum SequenceWindowsConfiguration {
    public static let operatorID="kwip.data.sequenceWindows"
    public static let defaults:[String:JSONValue]=["mode":.string("table"),"tokenColumn":.string("tokens"),"scope":.string("row"),"documentColumn":.string(""),"orderColumn":.string(""),"direction":.string("ascending"),"contextLength":128,"stride":128,"incomplete":.string("drop"),"paddingID":1,"inputColumn":.string("input_ids"),"targetColumn":.string("target_ids"),"maskColumn":.string("loss_mask"),"lengthColumn":.string("valid_length")]
    public static func validate(_ c:[String:JSONValue],schema:[SequenceWindowColumn]=[])throws {
        guard Set(c.keys)==Set(defaults.keys) else {throw DataFlowContractError("Sequence windows requires its exact configuration fields.")}
        for (key,options) in [("mode",["table","partitions"]),("scope",["row","document"]),("direction",["ascending","descending"]),("incomplete",["drop","pad"])] {
            guard let value=c[key]?.stringValue,options.contains(value) else {throw DataFlowContractError("Choose a valid Sequence windows \(key).")}
        }
        for (key,range) in [("contextLength",1...4096),("stride",1...65536),("paddingID",0...65535)] {
            guard case .integer(let value)=c[key],range.contains(Int(value)) else {throw DataFlowContractError("\(key) is outside its declared bounds.")}
        }
        for key in ["tokenColumn","documentColumn","orderColumn","inputColumn","targetColumn","maskColumn","lengthColumn"] {
            guard let name=c[key]?.stringValue,name.count<=200,!name.contains("\0"),["documentColumn","orderColumn"].contains(key) || !name.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty else {throw DataFlowContractError("Choose bounded nonblank sequence column names.")}
        }
        let outputs=["inputColumn","targetColumn","maskColumn","lengthColumn"].compactMap {c[$0]?.stringValue}
        guard Set(outputs).count==4,Set(outputs).isDisjoint(with:["window","document_id","start","parent_id"]) else {throw DataFlowContractError("Sequence output names must be distinct from metadata and one another.")}
        if c["scope"] == .string("document") {
            for key in ["documentColumn","orderColumn"] {guard let name=c[key]?.stringValue,!name.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty else {throw DataFlowContractError("Select document and order columns for grouped sequences.")}}
        }
        if !schema.isEmpty {
            guard schema.contains(where:{$0.name==c["tokenColumn"]?.stringValue && $0.dtype=="int64Sequence"}) else {throw DataFlowContractError("Select an actual integer token-sequence column.")}
            if c["scope"] == .string("document") {for key in ["documentColumn","orderColumn"] {guard schema.contains(where:{$0.name==c[key]?.stringValue && ["string","int64"].contains($0.dtype)}) else {throw DataFlowContractError("Document and order columns need text or integers.")}}}
        }
    }
    public static func validateConnectedPorts(_ c:[String:JSONValue],inputs:Set<String>,outputs:Set<String>)throws {
        try validate(c)
        let ports:Set<String>=c["mode"] == .string("table") ? ["table"]:["training","validation","test"]
        guard inputs.isSubset(of:ports),outputs.isSubset(of:ports) else {throw DataFlowContractError("Disconnect the existing Table or partition connections before changing Inputs. Your connections and saved configuration are preserved.")}
    }
    public static func columns(_ c:[String:JSONValue])->[SequenceWindowColumn] {
        guard (try? validate(c)) != nil else {return []}
        var result:[SequenceWindowColumn]=[("window","int64"),("document_id","string"),("start","int64"),("parent_id","string"),(c["inputColumn"]!.stringValue!,"int64Sequence"),(c["targetColumn"]!.stringValue!,"int64Sequence"),(c["lengthColumn"]!.stringValue!,"int64")].map {.init(name:$0.0,dtype:$0.1,nullable:false)}
        if c["incomplete"] == .string("pad") {result.append(.init(name:c["maskColumn"]!.stringValue!,dtype:"int64Sequence",nullable:false))}
        return result
    }
}
