import Foundation

/// Shared admission hook for local execution. The free app requires no account,
/// Keychain grant, device registration or commerce request.
extension WorkspaceStore {
    @discardableResult func allowExecution()->Bool {
        do {try executionAdmission();return true}
        catch {statusMessage=error.localizedDescription;return false}
    }
}
