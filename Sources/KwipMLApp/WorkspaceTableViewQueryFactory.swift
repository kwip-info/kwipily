import Foundation
import KwipMLCore

extension WorkspaceStore {
    /// The caller supplies a verified immutable history reference and selection
    /// lifetime. The installed runtime executes only the bounded index query.
    func makeTableViewQuerySession(artifact: TableViewQuerySession.Artifact,
                                  projectID: String,
                                  accepts: @escaping () -> Bool) -> TableViewQuerySession? {
        guard activeProjectID == projectID, let runtime,
              let policy = try? executionPolicy(for: projectID) else { return nil }
        weak var created: TableViewQuerySession?
        let reservation = WorkflowExecutionReservation(acquire: { [weak self] in
            guard let self, self.activeProjectID == projectID, accepts(), !self.isBusy else { return nil }
            self.cancelDataPreparation = { [weak created] in created?.cancel() }
            let token = self.dataPreparationReservationID
            self.isPreparingData = true
            return token
        }, release: { [weak self] token in
            guard let self, self.dataPreparationReservationID == token else { return }
            self.isPreparingData = false
            self.cancelDataPreparation = nil
        })
        let session = TableViewQuerySession(artifact: artifact, workspace: workspaceDirectory,
            accountingRoot: applicationDirectory, projectID: projectID, runtime: runtime,
            storageLimitBytes: policy.storageLimitBytes, reservation: reservation,
            accepts: { [weak self] in self?.activeProjectID == projectID && accepts() })
        created = session
        return session
    }
}
