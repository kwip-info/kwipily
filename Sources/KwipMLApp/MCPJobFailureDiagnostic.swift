import KwipMLCore

/// Public job diagnostics never serialize backend prose, row counts, labels,
/// paths, values or arbitrary worker codes. Attribution is graph-verified first.
enum MCPJobFailureDiagnostic {
    static func execution(_ message:String?)->[String:JSONValue] {
        var value=project(nil,graph:DataFlowDocument(nodes:[],edges:[]))
        // Exact match only: worker prose may contain source values or private paths.
        if message == "Mapped integer values require explicit int64 columns; floating values require numeric columns." {
            value["code"] = .string("mapped_column_types_required")
            value["message"] = .string("Train mappings require declared numeric columns. Add or update Columns before Train: integer inputs need int64; floating inputs need numeric columns. Check the input and target mappings before running again.")
        }
        return value
    }
    static func admission(_ message:String)->[String:JSONValue] {
        var value:[String:JSONValue] = ["isError":.bool(true),"code":.string("run_refused"),"message":.string("The app refused this run. Inspect native component Diagnostics and required file/folder selections before trying again.")]
        if ["Choose Save model’s destination before running this branch.","Choose Save model’s destination before running."].contains(message) {
            value["code"] = .string("model_destination_required")
            value["message"] = .string("Choose a parent folder for Save model in the app, then save the project before running. MCP cannot grant filesystem access or select a private path.")
        }
        return value
    }

    static func project(_ detail:DataFlowFailureDetail?,graph:DataFlowDocument)->[String:JSONValue] {
        var value:[String:JSONValue]=["code":.string("execution_failed"),"message":.string("The operation failed. Inspect the native component Diagnostics for details."),"retry":.string("Do not retry external writes until their delivery receipts are checked; a failed run may have committed earlier writes.")]
        guard let detail,let node=graph.nodes.first(where:{$0.id==detail.nodeID && $0.operatorID==detail.operatorID}) else{return value}
        value["component"] = .object(["domain":.string("data"),"localID":.string(node.id)])
        value["operatorID"] = .string(node.operatorID)
        if detail.code=="expression_numeric_required",node.operatorID=="kwip.data.formula" {
            value["code"] = .string("expression_numeric_required")
            value["message"] = .string("Formula arithmetic requires numeric inputs. Cast explicitly for arithmetic, or choose a documented text-composition operation for strings. Inspect the input schema and Formula configuration before running again.")
        }
        if node.operatorID=="kwip.data.sourceCSV",let stage=WorkflowAPICaptureStage.allCases.first(where:{$0.code==detail.code}) {
            value["code"] = .string(stage.code)
            value["message"] = .string(stage.message+" Inspect Table Input Diagnostics and the private connection before retrying.")
        }
        return value
    }
}
