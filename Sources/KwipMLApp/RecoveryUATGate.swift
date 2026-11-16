#if KWIPML_RECOVERY_UAT
import Foundation
import Combine

/// Deliberately absent from ordinary builds. One-shot native crash-test boundary.
@MainActor final class RecoveryUATGate: ObservableObject {
    static let shared = RecoveryUATGate()
    enum Stage: String, CaseIterable {
        case beforeWorkerSpawn = "Before worker spawn"
        case duringPreparation = "During preparation"
        case betweenVariants = "Between variants"
    }
    struct Identity: Equatable {
        let groupID: String
        let runID: String
        let ordinal: Int
        let nonce: String?
    }
    struct Reached {
        let token: UUID
        let stage: Stage
        let identity: Identity
    }
    @Published private(set) var armed: Stage?
    @Published private(set) var reached: Reached?
    private var resumeAction: (() -> Void)?
    private var cancelAction: (() -> Void)?
    private var timeout: Task<Void, Never>?

    func arm(_ stage: Stage) {
        guard reached == nil else { return }
        armed = stage
    }
    func disarm() { armed = nil }
    func hold(_ stage: Stage, identity: Identity, resume: @escaping () -> Void, cancel: @escaping () -> Void) -> UUID? {
        guard armed == stage, reached == nil else { return nil }
        let token = UUID()
        armed = nil
        resumeAction = resume; cancelAction = cancel
        reached = Reached(token: token, stage: stage, identity: identity)
        timeout = Task { [weak self] in
            do { try await Task.sleep(nanoseconds: 300_000_000_000) }
            catch { return }
            self?.expire(token)
        }
        return token
    }
    func resume() {
        guard let reached else { return }
        let action = resumeAction
        dismiss(reached.token)
        action?()
    }
    func cancel(_ token: UUID) {
        guard reached?.token == token else { return }
        let action = cancelAction
        dismiss(token)
        action?()
    }
    func expire(_ token: UUID) { cancel(token) }
    func dismiss(_ token: UUID) {
        guard reached?.token == token else { return }
        timeout?.cancel(); timeout = nil
        resumeAction = nil; cancelAction = nil
        reached = nil
    }
}
#endif
