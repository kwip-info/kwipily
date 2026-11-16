import Foundation
import KwipMLCore

enum DataEngineeringConfiguration {
    static let operatorIDs:Set<String>=["filterExpression","calculate","regexMatch","regexExtract","regexReplace","outputDelimited"].reduce(into:[]) {$0.insert("kwip.data."+$1)}
    static func validate(_ config:[String:JSONValue],definition:DataFlowOperatorDefinition,schema:[DataColumnSchema])throws->[String:JSONValue] {
        guard operatorIDs.contains(definition.id) else {throw DataFlowContractError("This editor cannot configure that operator.")}
        try definition.validateConfiguration(config)
        let available=Set(schema.map(\.name))
        func requireColumn(_ name:String)throws {
            guard !name.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty, available.isEmpty || available.contains(name) else {throw DataFlowContractError("Choose a column from the connected table.")}
        }
        func requireOutput(_ name:String)throws {
            guard !name.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,!name.contains("\0") else {throw DataFlowContractError("Give the output column a name.")}
            if available.contains(name),config["replaceExisting"] != .bool(true) {throw DataFlowContractError("That column already exists. Choose another name or explicitly allow replacing existing columns.")}
        }
        if let column=config["column"]?.data02String {try requireColumn(column)}
        if let output=config["outputColumn"]?.data02String {try requireOutput(output)}
        if let groups=config["groups"]?.data02Array {
            var names=Set<String>()
            for group in groups {
                guard let name=group.data02Object?["output"]?.data02String,names.insert(name).inserted else {throw DataFlowContractError("Each extracted column needs a unique output name.")}
                try requireOutput(name)
            }
        }
        if definition.id == "kwip.data.outputDelimited", !APITableSettings.isExternal(config) {
            let filename=config["filename"]?.data02String ?? ""
            guard !filename.isEmpty,filename != ".",filename != "..",!filename.contains("/"),!filename.contains("\\"),!filename.contains("\0") else {throw DataFlowContractError("Use a filename only. Choose its folder with Choose destination.")}
            for column in config["columns"]?.data02Array ?? [] {if let name=column.data02String {try requireColumn(name)}}
        }
        return config
    }
}

private extension JSONValue {
    var data02String:String? {if case .string(let value)=self {return value};return nil}
    var data02Array:[JSONValue]? {if case .array(let value)=self {return value};return nil}
    var data02Object:[String:JSONValue]? {if case .object(let value)=self {return value};return nil}
}
