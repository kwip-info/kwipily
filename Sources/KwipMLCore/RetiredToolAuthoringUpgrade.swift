import Foundation

/// Current editable source only. Captured History bytes are never upgraded here.
public enum RetiredToolAuthoringUpgrade {
    public struct RepairRequired: Error, LocalizedError {
        public let detail: String
        public var errorDescription: String? {
            "This workflow needs repair before it can run with current tools. " + detail +
            " The original project is unchanged. Replace the marked retired components and review their connections."
        }
    }
    public static func requiresUpgrade(_ document: WorkflowDocument) -> Bool {
        RetiredTrainingRecipeMigration.requiresUpgrade(document) || RetiredDataOperatorMigration.requiresUpgrade(document) || WorkflowTrainRunSetup.requiresUpgrade(document) || WorkflowTrainMapping.requiresUpgrade(document)
    }
    public static func plan(_ document: WorkflowDocument) throws -> WorkflowDocument {
        try WorkflowCodec.validate(document)
        guard requiresUpgrade(document) else { return document }
        do {
            let training = try RetiredTrainingRecipeMigration.migrate(document)
            let runSettings=try WorkflowTrainRunSetup.migrate(training)
            return try WorkflowTrainMapping.migrate(RetiredDataOperatorMigration.migrate(runSettings))
        } catch {
            // Neither pure stage writes a source. Failure cannot publish half an upgrade.
            throw RepairRequired(detail: error.localizedDescription)
        }
    }
}
