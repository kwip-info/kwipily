import Foundation

/// Reads the worker-owned machine-readable registry; config schemas are not copied into Swift switches.
public struct DataFlowOperatorRegistry:Codable,Equatable {
    public var schemaVersion:Int
    public var packageID:String
    public var operators:[DataFlowOperatorDefinition]
    public func operatorDefinition(id:String)->DataFlowOperatorDefinition? {operators.first {$0.id==id}}
    public static func load(from url:URL)throws->Self {
        let data=try Data(contentsOf:url)
        guard data.count<=2_000_000 else {throw DataFlowContractError("Operator registry exceeds 2 MB.")}
        let result=try JSONDecoder().decode(Self.self,from:data)
        guard result.schemaVersion==1,!result.packageID.isEmpty,result.operators.count<=512,Set(result.operators.map(\.id)).count==result.operators.count else {throw DataFlowContractError("Unsupported or duplicate operator registry.")}
        for op in result.operators {
            guard !op.id.isEmpty,op.id.count<=128,Set(op.inputs.map(\.id)).count==op.inputs.count,Set(op.outputs.map(\.id)).count==op.outputs.count else {throw DataFlowContractError("Invalid operator or port identity.")}
            guard op.canvasRole == nil || op.canvasRole == "sink" else {throw DataFlowContractError("Unsupported operator canvas role.")}
            guard case .object=op.configSchema else {throw DataFlowContractError("Operator configuration schema must be an object.")}
        }
        return result
    }
}
public struct DataFlowOperatorDefinition:Codable,Equatable,Identifiable {
    public var id:String
    public var version:ContractVersion
    public var title:String
    public var backend:String
    public var device:String
    public var inputs:[DataFlowOperatorPort]
    public var outputs:[DataFlowOperatorPort]
    public var configSchema:JSONValue
    public var defaultConfig:[String:JSONValue]
    public var fitBoundary:String
    /// Presentation only. Execution outputs remain stable for retained runs and legacy wires.
    public var canvasRole:String? = nil
    public func configured(_ values:[String:JSONValue])->Self {
        if id == "kwip.data.union", case .array(let ports) = values["inputs"], let template = inputs.first {
            var result=self
            result.inputs=ports.compactMap {if case .string(let id)=$0 {var port=template;port.id=id;return port};return nil}
            return result
        }
        if id=="kwip.data.sequenceWindows" {
            var result=self;let table=values["mode"] == .string("table")
            result.inputs=inputs.filter {table ? $0.id=="table":$0.id != "table"}.map {port in var p=port;p.required=table || p.id=="training";return p}
            result.outputs=outputs.filter {table ? $0.id=="table":$0.id != "table"};return result
        }
        if id=="kwip.data.textEncode" {
            var result=self
            let supplied=values["vocabularySource"] == .string("supplied")
            result.inputs=inputs.filter {supplied ? $0.id=="table":$0.id != "table"}.map {port in var p=port;p.required=supplied || p.id=="training";return p}
            result.outputs=outputs.filter {supplied ? ["table","state","vocabulary"].contains($0.id):$0.id != "table"}
            return result
        }
        guard id=="kwip.data.preprocess" else {return self}
        var result=self
        let frozen=values["mode"] == .string("frozen")
        let table=values["mode"] == .string("table") || frozen
        result.inputs=inputs.filter {table ? ($0.id=="table" || (frozen && $0.id=="model")):!["table","model"].contains($0.id)}.map {port in var p=port;p.required=table || p.id=="training";return p}
        result.outputs=outputs.filter {table ? $0.id=="table":$0.id != "table"}
        return result
    }
    public func executionOutputs(_ values:[String:JSONValue],connectedInputs:Set<String>)->[DataFlowOperatorPort] {
        let ports=configured(values).outputs
        if id=="kwip.data.sequenceWindows",values["mode"] == .string("partitions") {return ports.filter {!["validation","test"].contains($0.id) || connectedInputs.contains($0.id)}}
        if id=="kwip.data.textEncode",values["vocabularySource"] != .string("supplied") {return ports.filter {!["validation","test"].contains($0.id) || connectedInputs.contains($0.id)}}
        guard id=="kwip.data.preprocess",values["mode"] != .string("table") else {return ports}
        return ports.filter {!["validation","test"].contains($0.id) || connectedInputs.contains($0.id)}
    }
    public func configuredForExecution(_ values:[String:JSONValue],connectedInputs:Set<String>)->Self {
        var result=configured(values);result.outputs=executionOutputs(values,connectedInputs:connectedInputs);return result
    }
    public var isCanvasSink:Bool {canvasRole == "sink" || id == "kwip.data.view"}
    public var authoringOutputs:[DataFlowOperatorPort] {isCanvasSink ? []:outputs.filter {$0.inspectionOnly != true}}
}
public struct DataFlowOperatorPort:Codable,Equatable,Identifiable {
    public var inspectionOnly:Bool? = nil
    public var id:String
    public var kind:String
    public var partition:String?
    public var stateKind:String?
    public var required:Bool
    public var maximumConnections:Int?
}
