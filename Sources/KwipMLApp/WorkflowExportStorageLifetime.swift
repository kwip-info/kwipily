import Foundation
import KwipMLCore

/// Explicitly attached only by an export writer. A shared review-package value
/// retains this lifetime after worker exit; ordinary persisted readers do not.
final class WorkflowExportStorageLifetime {
    private let directory: WorkflowFrozenSnapshotDirectory
    private var reservation: WorkflowStorageAdmission.Reservation?
    private let lock = NSLock()
    private var removed = false

    init(directory: URL, reservation: WorkflowStorageAdmission.Reservation?) throws {
        do {
            self.directory = try WorkflowFrozenSnapshotDirectory(directory: directory, removeOnDeinit: false)
            self.reservation = reservation
        } catch {
            reservation?.close()
            throw error
        }
    }

    /// Retire while bound authority pins still exist. Active owners or workers
    /// preserve the package and allowance; explicit removal can retry after exit.
    /// Retained payload remains counted until its pinned directory is removed.
    func remove() {
        lock.lock(); defer { lock.unlock() }
        guard !removed else { return }
        do { try reservation?.retire() } catch { return }
        removed = true
        directory.remove()
        reservation?.close(); reservation = nil
    }
    deinit { remove() }
}
