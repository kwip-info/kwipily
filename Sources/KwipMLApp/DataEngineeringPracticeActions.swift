import Foundation
import KwipMLCore

extension WorkspaceStore {
    @discardableResult func loadDataEngineeringPracticeExample(_ id: String) -> Bool {
        guard DataEngineeringPractice.isExample(id) else { statusMessage = "That data practice is unavailable in this build."; return false }
        return loadPreparedWorkflowExample(device: "cpu", kind: .component(id))
    }
}
