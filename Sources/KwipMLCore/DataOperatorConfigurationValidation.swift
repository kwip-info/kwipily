import Foundation

extension DataFlowOperatorDefinition {
    /// Shared configuration schema authority used by native editors and compilation.
    public func validateConfiguration(_ configuration:[String:JSONValue]) throws {
        try WorkflowSchema.check(.object(configuration),schema:configSchema)
        if id=="kwip.data.join" {try JoinProjectionContract.validate(configuration)}
    }
}

/// Optional projection settings form one contract; legacy joins omit both.
public enum JoinProjectionContract {
    public static func validate(_ configuration:[String:JSONValue]) throws {
        guard (configuration["outputColumns"] != nil) == (configuration["newColumns"] != nil) else {
            throw DataFlowContractError("Supply output columns and new-column policy together.")
        }
    }
}
