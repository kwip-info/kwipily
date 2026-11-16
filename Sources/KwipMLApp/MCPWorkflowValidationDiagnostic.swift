import KwipMLCore

/// Only finite compiler categories and identities verified in the authorized
/// captured document cross MCP. Compiler prose and model issues stay native.
enum MCPWorkflowValidationDiagnostic {
    static func project(_ error:Error,document:WorkflowDocument?,ports:(WorkflowNode)throws->Set<String>)->[String:JSONValue] {
        var result:[String:JSONValue]=["isError":.bool(true),"valid":.bool(false),"code":.string("workflow_validation_failed"),"message":.string("The authoring plan could not be validated. Inspect native component validation before running."),"scope":.string("Authoring validation only; no execution, model compatibility, private-resource readiness or successful output is proved.")]
        guard let document,let failure=error as? WorkflowCompilerError else{return result}
        let guidance:String
        switch failure.code {
        case .invalidDocument:guidance="The workflow document is invalid. Inspect the canvas structure and native validation."
        case .invalidSelection:guidance="Choose a supported explicit terminal from the current workflow."
        case .cycle:guidance="Remove the dependency cycle before validating this execution plan."
        case .requiredInput:guidance="Connect the required input using the component's published port contract."
        case .invalidPort:guidance="Inspect current component ports and replace the invalid connection."
        case .portKind:guidance="Connect ports with compatible data or model kinds; inspect their published contracts."
        case .configuration:guidance="Review this component's complete configuration against current catalog fields and native validation."
        case .unsupportedOperator:guidance="The selected plan contains an unsupported operator. Inspect current catalog capabilities."
        case .lineage:guidance="Check connected partition roles, source, split and fitted-state lineage. Preserve training and held-out boundaries."
        case .unresolvedDraft:guidance="Apply or discard pending workflow drafts, then validate again."
        case .mixedModelVersions:guidance="Use compatible model-definition versions throughout this plan."
        case .invalidModel:guidance="Inspect native model validation and correct the model graph or mappings."
        }
        result["reason"] = .string(failure.code.rawValue);result["message"] = .string(guidance)
        if let address=failure.node,let node=document.nodes.first(where:{$0.address==address}) {
            result["component"] = .object(["domain":.string(address.domain.rawValue),"localID":.string(address.localID)])
            if let port=failure.port,(try? ports(node).contains(port)) == true {result["port"] = .string(port)}
        }
        return result
    }
}
