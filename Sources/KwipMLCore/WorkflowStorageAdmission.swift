import Foundation
import Darwin

/// Admission accounting for managed execution storage only. This does not limit the
/// filesystem or delete retained work; writers must honor their declared peak bound.
public final class WorkflowStorageAdmission {
    public enum Namespace: String, Codable, CaseIterable {
        case workflowOperations = "workflow-operations", workflowRuns = "workflow-runs"
        case modelAssets = "model-assets", modelDownloadStaging = "model-download-staging"
        case ocrPacks = "ocr-packs"
        case workflowExperiments = "workflow-experiments", workflowEvaluations = "workflow-evaluations"
        case workflowEvaluationExports = "workflow-evaluation-exports"
        case runs, previews, dataflowRuns = "dataflow/runs", sourceSnapshots = "source-snapshots"
        case previewStaging = "preview-staging", preparationStaging = "preparation-staging"
        case sourceInspectionStaging = "source-inspection-staging", preparationInspectionStaging = "preparation-inspection-staging"
        case modelTrainingStaging = "model-training-staging", preparationSourceStaging = "preparation-source-staging"
        case evaluationStaging = "evaluation-staging"
        case trainedExportStaging = "trained-export-staging"
        case checkpointStaging = "checkpoint-staging"
        case workflowSourceStaging = "workflow-source-staging"
        case databaseSourceSnapshots = "database-source-snapshots"
        case databaseOutputRecords = "database-output-records"
        case webhooks
        fileprivate var parts: [String] { rawValue.split(separator: "/").map(String.init) }
        fileprivate var scopeDepth: Int {
            switch self {
            case .modelAssets, .workflowOperations, .workflowRuns, .workflowExperiments, .workflowEvaluations, .workflowEvaluationExports: return 2
            default: return 1
            }
        }
    }
    public struct Scope: Codable, Equatable {
        public let namespace: Namespace
        public let components: [String]
        public var relativeComponents: [String] { namespace.parts + components }
        public init(namespace: Namespace, components: [String]) throws {
            guard namespace != .sourceSnapshots, namespace != .previews, components.count == namespace.scopeDepth,
                  components.allSatisfy({ UUID(uuidString: $0)?.uuidString == $0 }) else { throw Failure.invalidScope }
            self.namespace = namespace; self.components = components
        }
        fileprivate func validate() throws { _ = try Self(namespace: namespace, components: components) }
    }
    public struct ScanLimits {
        public let maximumEntries: Int
        public let maximumDepth: Int
        public init(maximumEntries: Int = 100_000, maximumDepth: Int = 32) {
            self.maximumEntries = maximumEntries; self.maximumDepth = maximumDepth
        }
    }
    public struct Snapshot: Equatable {
        public let retainedBytes: Int64
        public let reservedGrowthBytes: Int64
        public var committedBytes: Int64 { retainedBytes + reservedGrowthBytes }
    }
    public enum Failure: Error, LocalizedError {
        case invalidScope, unsafeStorage, scanLimit, overlappingScope, invalidLedger
        case capacity(limitBytes: Int64, demandBytes: Int64)
        public var errorDescription: String? {
            switch self {
            case .invalidScope: return "Storage reservation requires a bounded, private execution scope."
            case .unsafeStorage: return "Managed execution storage contains an unsafe or changed filesystem entry."
            case .scanLimit: return "Managed execution storage exceeds its bounded accounting scan."
            case .capacity(let limit, let demand): return "Managed workspace storage requires \(demand) bytes including retained outputs and reserved growth; the captured allowance is \(limit) bytes. Increase Managed storage in Settings before retrying. Existing work was retained."
            case .overlappingScope: return "Another live reservation owns this execution storage scope."
            case .invalidLedger: return "Execution storage reservation metadata is invalid or exceeds its bound."
            }
        }
    }
    public struct Authority: Codable, Equatable {
        public let scope: Scope
        public let directoryIdentity: WorkflowFileIdentity
        public let ownerLockIdentity: WorkflowFileIdentity
        public let activityLockIdentity: WorkflowFileIdentity
        public init(scope: Scope, directoryIdentity: WorkflowFileIdentity, ownerLockIdentity: WorkflowFileIdentity, activityLockIdentity: WorkflowFileIdentity) {
            self.scope = scope; self.directoryIdentity = directoryIdentity
            self.ownerLockIdentity = ownerLockIdentity; self.activityLockIdentity = activityLockIdentity
        }
        fileprivate func validate() throws {
            try scope.validate()
            guard [.workflowOperations, .workflowExperiments, .workflowEvaluations, .workflowEvaluationExports].contains(scope.namespace) else { throw Failure.invalidScope }
        }
    }
    /// Staging authority remains live through native finalization. A bound worker's
    /// existing owner/activity leases independently preserve its allowance after close.
    public final class Reservation {
        public let id: String
        private let admission: WorkflowStorageAdmission
        private var lease: WorkflowFileLease?
        private let lock = NSLock()
        fileprivate init(id: String, lease: WorkflowFileLease, admission: WorkflowStorageAdmission) {
            self.id = id; self.lease = lease; self.admission = admission
        }
        public func bind(to authority: Authority) throws {
            lock.lock(); defer { lock.unlock() }
            guard let lease else { throw Failure.invalidLedger }
            try lease.checkIdentity()
            try admission.bind(id: id, lease: lease, authority: authority)
        }
        /// Admit the next sequential unit while retaining this staging lease and
        /// bound owner. Completed scopes stay charged as retained bytes. A worker
        /// must be idle before any allowance or scope can change.
        public func revise(scopes: [Scope], peakBytes: Int64, limitBytes: Int64) throws {
            lock.lock(); defer { lock.unlock() }
            guard let lease else { throw Failure.invalidLedger }
            try lease.checkIdentity()
            try admission.revise(id: id, lease: lease, scopes: scopes, peakBytes: peakBytes, limitBytes: limitBytes)
        }
        /// Keep the full allowance while leaving a completed phase's authority.
        /// Failure leaves the record bound and funded for terminal reporting.
        public func prepareNextPhase(limitBytes: Int64) throws {
            lock.lock(); defer { lock.unlock() }
            guard let lease else { throw Failure.invalidLedger }
            try admission.prepareNextPhase(id: id, lease: lease, limitBytes: limitBytes)
        }
        /// Before deleting a bound authority directory, retire its ledger entry
        /// while its pinned owner/activity locks still exist for the inactivity proof.
        public func retire() throws {
            lock.lock(); defer { lock.unlock() }
            lease?.close(); lease = nil
            try admission.transaction { _, ledger in
                guard !ledger.reservations.contains(where: { $0.id == self.id }) else { throw WorkflowFileLeaseError.busy }
            }
        }
        public func close() { lock.lock(); defer { lock.unlock() }; lease?.close(); lease = nil }
        deinit { close() }
    }
    private struct Record: Codable {
        let id: String
        let scopes: [Scope]
        let peakBytes: Int64
        let directoryIdentity: WorkflowFileIdentity
        let lockIdentity: WorkflowFileIdentity
        var authority: Authority? = nil
    }
    private struct Ledger: Codable { var schemaVersion = 1; var reservations: [Record] = [] }
    private let root: URL
    var beforeAccountingMemberOpen: ((URL) throws -> Void)?
    private let accountingRoot: URL?
    private let accountingIdentity: WorkflowFileIdentity?
    private let identity: WorkflowFileIdentity
    private let limits: ScanLimits
    private static let ledgerLimit = 2_000_000
    private static let recordLimit = 4096

    public convenience init(workspace: URL, accountingRoot: URL? = nil, limits: ScanLimits = .init()) throws {
        try self.init(workspace: workspace, accountingRoot: accountingRoot, limits: limits, expectedIdentity: nil)
    }
    private init(workspace: URL, accountingRoot: URL?, limits: ScanLimits, expectedIdentity: WorkflowFileIdentity?) throws {
        guard workspace.isFileURL, !workspace.path.contains("\0"), (1...1_000_000).contains(limits.maximumEntries),
              (1...64).contains(limits.maximumDepth) else { throw Failure.invalidScope }
        let root: URL
        if expectedIdentity != nil {
            // A peer was enumerated under the domain gate. Never resolve a replacement
            // link, create its directory, or publish controls before checking its pin.
            root = workspace
        } else {
            try FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: true)
            guard let pointer = realpath(workspace.path, nil) else { throw Failure.unsafeStorage }
            root = URL(fileURLWithPath: String(cString: pointer)); free(pointer)
        }
        let fd = try Self.openDirectory(root)
        defer { Darwin.close(fd) }
        if let expectedIdentity { guard try Self.directoryIdentity(fd) == expectedIdentity else { throw Failure.unsafeStorage } }
        if let accountingRoot {
            guard accountingRoot.isFileURL, !accountingRoot.path.contains("\0"), let globalPointer = realpath(accountingRoot.path, nil) else { throw Failure.unsafeStorage }
            let global = URL(fileURLWithPath: String(cString: globalPointer)); free(globalPointer)
            let expectedParent = global.appendingPathComponent("external-projects")
            guard root.path == global.path || (root.deletingLastPathComponent().path == expectedParent.path &&
                UUID(uuidString: root.lastPathComponent)?.uuidString == root.lastPathComponent) else { throw Failure.unsafeStorage }
            let globalFD = try Self.openDirectory(global)
            defer { Darwin.close(globalFD) }
            if root.path != global.path {
                let members = try Self.child(globalFD, "external-projects", create: false)
                defer { Darwin.close(members) }
                let member = try Self.child(members, root.lastPathComponent, create: false)
                defer { Darwin.close(member) }
                guard try Self.directoryIdentity(member) == Self.directoryIdentity(fd) else { throw Failure.unsafeStorage }
            }
            self.accountingRoot = global
            self.accountingIdentity = try Self.directoryIdentity(globalFD)
            // Establish the shared control without changing any local artifact path.
            let globalControl = try Self.child(globalFD, "storage-accounting", create: true)
            Darwin.close(globalControl)
        } else { self.accountingRoot = nil; self.accountingIdentity = nil }
        self.root = root; self.identity = try Self.directoryIdentity(fd); self.limits = limits
        let control = try Self.child(fd, "storage-accounting", create: true)
        defer { Darwin.close(control) }
        let lease = try WorkflowFileLease(directory: root.appendingPathComponent("storage-accounting"), name: "owner.lock", mode: .exclusive, create: true, requireLocalAPFS: true)
        defer { lease.close() }
        let reservations = try Self.child(control, "reservations", create: true)
        Darwin.close(reservations)
        try lease.checkIdentity()
    }
    public func snapshot() throws -> Snapshot {
        try transaction { _, ledger in try self.measure(ledger.reservations) }
    }
    public func reserve(scopes: [Scope], peakBytes: Int64, limitBytes: Int64) throws -> Reservation {
        guard !scopes.isEmpty, scopes.count <= 16, peakBytes > 0, limitBytes >= 0 else { throw Failure.invalidScope }
        for scope in scopes { try scope.validate() }
        try Self.requireDisjoint(scopes)
        return try transaction { control, ledger in
            for record in ledger.reservations {
                for proposed in scopes where record.scopes.contains(where: { Self.overlaps($0, proposed) }) { throw Failure.overlappingScope }
            }
            // Proposed scopes participate in the same disjoint accounting pass as live
            // scopes; never subtract a later size from an earlier retained-byte scan.
            let proposed = try self.measure(ledger.reservations, additional: (scopes, peakBytes))
            guard proposed.committedBytes <= limitBytes else { throw Failure.capacity(limitBytes: limitBytes, demandBytes: proposed.committedBytes) }
            guard ledger.reservations.count < Self.recordLimit else { throw Failure.invalidLedger }
            let parent = try Self.child(control, "reservations", create: false)
            defer { Darwin.close(parent) }
            guard try Self.names(parent, maximum: Self.recordLimit).count < Self.recordLimit else { throw Failure.invalidLedger }
            let id = UUID().uuidString
            let directory = try Self.child(parent, id, create: true, exclusive: true)
            defer { Darwin.close(directory) }
            let url = self.root.appendingPathComponent("storage-accounting/reservations/" + id)
            let lease = try WorkflowFileLease(directory: url, name: "owner.lock", mode: .exclusive, create: true, createExclusive: true, requireLocalAPFS: true)
            do {
                let record = Record(id: id, scopes: scopes, peakBytes: peakBytes,
                    directoryIdentity: try Self.directoryIdentity(directory), lockIdentity: lease.identity)
                ledger.reservations.append(record)
                try self.save(ledger, control: control)
                return Reservation(id: id, lease: lease, admission: self)
            } catch { lease.close(); throw error }
        }
    }
    private func transaction<T>(_ body: (Int32, inout Ledger) throws -> T) throws -> T {
        var globalGate: WorkflowFileLease?
        if let accountingRoot {
            let fd = try Self.openDirectory(accountingRoot)
            defer { Darwin.close(fd) }
            guard try Self.directoryIdentity(fd) == accountingIdentity else { throw Failure.unsafeStorage }
            globalGate = try WorkflowFileLease(directory: accountingRoot.appendingPathComponent("storage-accounting"),
                name: "activity.lock", mode: .exclusive, create: true, requireLocalAPFS: true)
        }
        defer { globalGate?.close() }
        let fd = try Self.openDirectory(root); defer { Darwin.close(fd) }
        guard try Self.directoryIdentity(fd) == identity else { throw Failure.unsafeStorage }
        let control = try Self.child(fd, "storage-accounting", create: false); defer { Darwin.close(control) }
        let gate = try WorkflowFileLease(directory: root.appendingPathComponent("storage-accounting"), name: "owner.lock", mode: .exclusive, requireLocalAPFS: true)
        defer { gate.close() }
        guard gate.parentIdentity == (try Self.directoryIdentity(control)) else { throw Failure.unsafeStorage }
        try validateControl(control)
        var ledger = try load(control)
        var inactive: [(Record, [WorkflowFileLease])] = []
        defer { inactive.flatMap { $0.1 }.forEach { $0.close() } }
        for record in ledger.reservations {
            if let probes = try inactiveProbes(record) { inactive.append((record, probes)) }
        }
        if !inactive.isEmpty {
            let ids = Set(inactive.map { $0.0.id })
            ledger.reservations.removeAll { ids.contains($0.id) }
            // Persist first: a crash leaves an orphan control directory that the
            // same gate can safely reclaim, never a ledger pointing at removed pins.
            try save(ledger, control: control)
            for (record, _) in inactive { try reclaim(record: record, control: control) }
        }
        try reclaimOrphans(control: control, tracked: Set(ledger.reservations.map(\.id)))
        let result = try body(control, &ledger)
        try globalGate?.checkIdentity()
        try gate.checkIdentity()
        return result
    }
    private func bind(id: String, lease: WorkflowFileLease, authority: Authority) throws {
        try authority.validate()
        try transaction { control, ledger in
            guard let index = ledger.reservations.firstIndex(where: { $0.id == id }),
                  ledger.reservations[index].authority == nil,
                  ledger.reservations[index].scopes.contains(authority.scope),
                  ledger.reservations[index].lockIdentity == lease.identity,
                  ledger.reservations[index].directoryIdentity == lease.parentIdentity else { throw Failure.invalidLedger }
            try self.checkAuthority(authority)
            let url = self.authorityURL(authority)
            do {
                let probe = try WorkflowFileLease(directory: url, name: "owner.lock", mode: .shared, expectedIdentity: authority.ownerLockIdentity, requireLocalAPFS: true)
                probe.close()
                throw Failure.invalidLedger
            } catch WorkflowFileLeaseError.busy { /* Exclusive native owner is required. */ }
            try lease.checkIdentity()
            ledger.reservations[index].authority = authority
            try self.save(ledger, control: control)
        }
    }
    private func revise(id: String, lease: WorkflowFileLease, scopes: [Scope], peakBytes: Int64, limitBytes: Int64) throws {
        guard !scopes.isEmpty, scopes.count <= 16, peakBytes > 0, limitBytes >= 0 else { throw Failure.invalidScope }
        for scope in scopes { try scope.validate() }
        try Self.requireDisjoint(scopes)
        try transaction { control, ledger in
            guard let index = ledger.reservations.firstIndex(where: { $0.id == id }),
                  ledger.reservations[index].lockIdentity == lease.identity,
                  ledger.reservations[index].directoryIdentity == lease.parentIdentity else { throw Failure.invalidLedger }
            let previous = ledger.reservations[index]
            var activity: WorkflowFileLease?
            defer { activity?.close() }
            if let authority = previous.authority {
                guard scopes.contains(authority.scope) else { throw Failure.invalidScope }
                try self.checkAuthority(authority)
                let url = self.authorityURL(authority)
                // A shared reader cannot substitute for the exclusive native owner.
                do {
                    let probe = try WorkflowFileLease(directory: url, name: "owner.lock", mode: .shared, expectedIdentity: authority.ownerLockIdentity, requireLocalAPFS: true)
                    probe.close(); throw Failure.unsafeStorage
                } catch WorkflowFileLeaseError.busy { /* Native owner remains live. */ }
                activity = try WorkflowFileLease(directory: url, name: "activity.lock", mode: .exclusive, expectedIdentity: authority.activityLockIdentity, requireLocalAPFS: true)
                try self.checkAuthority(authority)
            }
            let others = ledger.reservations.filter { $0.id != id }
            for record in others {
                for proposed in scopes where record.scopes.contains(where: { Self.overlaps($0, proposed) }) { throw Failure.overlappingScope }
            }
            let measured = try self.measure(others, additional: (scopes, peakBytes))
            guard measured.committedBytes <= limitBytes else { throw Failure.capacity(limitBytes: limitBytes, demandBytes: measured.committedBytes) }
            try lease.checkIdentity()
            try activity?.checkIdentity()
            if let authority = previous.authority {
                try self.checkAuthority(authority)
                do {
                    let probe = try WorkflowFileLease(directory: self.authorityURL(authority), name: "owner.lock", mode: .shared, expectedIdentity: authority.ownerLockIdentity, requireLocalAPFS: true)
                    probe.close(); throw Failure.unsafeStorage
                } catch WorkflowFileLeaseError.busy { /* Revalidate ownership after the scan. */ }
            }
            ledger.reservations[index] = Record(id: previous.id, scopes: scopes, peakBytes: peakBytes,
                directoryIdentity: previous.directoryIdentity, lockIdentity: previous.lockIdentity, authority: previous.authority)
            try self.save(ledger, control: control)
        }
    }
    private func prepareNextPhase(id: String, lease: WorkflowFileLease, limitBytes: Int64) throws {
        guard limitBytes >= 0 else { throw Failure.invalidScope }
        try lease.checkIdentity()
        try transaction { control, ledger in
            guard let index = ledger.reservations.firstIndex(where: { $0.id == id }),
                  ledger.reservations[index].lockIdentity == lease.identity,
                  ledger.reservations[index].directoryIdentity == lease.parentIdentity else { throw Failure.invalidLedger }
            var probes: [WorkflowFileLease] = []
            defer { probes.forEach { $0.close() } }
            let authority = ledger.reservations[index].authority
            if let authority {
                try self.checkAuthority(authority)
                let url = self.authorityURL(authority)
                probes.append(try WorkflowFileLease(directory: url, name: "owner.lock", mode: .shared, expectedIdentity: authority.ownerLockIdentity, requireLocalAPFS: true))
                probes.append(try WorkflowFileLease(directory: url, name: "activity.lock", mode: .exclusive, expectedIdentity: authority.activityLockIdentity, requireLocalAPFS: true))
            }
            let measured = try self.measure(ledger.reservations)
            guard measured.committedBytes <= limitBytes else { throw Failure.capacity(limitBytes: limitBytes, demandBytes: measured.committedBytes) }
            try lease.checkIdentity()
            for probe in probes { try probe.checkIdentity() }
            if let authority { try self.checkAuthority(authority) }
            ledger.reservations[index].authority = nil
            try self.save(ledger, control: control)
        }
    }
    private func authorityURL(_ authority: Authority) -> URL {
        authority.scope.relativeComponents.reduce(root) { $0.appendingPathComponent($1) }
    }
    private func checkAuthority(_ authority: Authority) throws {
        try authority.validate()
        let fd = try Self.openDirectory(authorityURL(authority)); defer { Darwin.close(fd) }
        guard try Self.directoryIdentity(fd) == authority.directoryIdentity else { throw Failure.unsafeStorage }
        for (name, expected) in [("owner.lock", authority.ownerLockIdentity), ("activity.lock", authority.activityLockIdentity)] {
            try Self.controlFile(fd, name: name, maximum: 0)
            var info = stat()
            guard fstatat(fd, name, &info, AT_SYMLINK_NOFOLLOW) == 0, Self.fileIdentity(info) == expected else { throw Failure.unsafeStorage }
        }
    }
    /// Returned probes remain held through ledger removal and control reclamation.
    /// Owner SH never impersonates native EX ownership; activity EX closes the
    /// handoff race only after native ownership is proven absent.
    private func inactiveProbes(_ record: Record) throws -> [WorkflowFileLease]? {
        let url = root.appendingPathComponent("storage-accounting/reservations/" + record.id)
        let fd = try Self.openDirectory(url); defer { Darwin.close(fd) }
        guard try Self.directoryIdentity(fd) == record.directoryIdentity else { throw Failure.unsafeStorage }
        var probes: [WorkflowFileLease] = []
        do {
            probes.append(try WorkflowFileLease(directory: url, name: "owner.lock", mode: .shared, expectedIdentity: record.lockIdentity, requireLocalAPFS: true))
            if let authority = record.authority {
                try checkAuthority(authority)
                let directory = authorityURL(authority)
                probes.append(try WorkflowFileLease(directory: directory, name: "owner.lock", mode: .shared, expectedIdentity: authority.ownerLockIdentity, requireLocalAPFS: true))
                probes.append(try WorkflowFileLease(directory: directory, name: "activity.lock", mode: .exclusive, expectedIdentity: authority.activityLockIdentity, requireLocalAPFS: true))
                try checkAuthority(authority)
            }
            return probes
        } catch WorkflowFileLeaseError.busy { probes.forEach { $0.close() }; return nil }
        catch { probes.forEach { $0.close() }; throw error }
    }
    private func reclaim(record: Record, control: Int32) throws {
        let parent = try Self.child(control, "reservations", create: false); defer { Darwin.close(parent) }
        let directory = try Self.child(parent, record.id, create: false); defer { Darwin.close(directory) }
        guard try Self.directoryIdentity(directory) == record.directoryIdentity,
              try Self.names(directory, maximum: 1) == ["owner.lock"] else { throw Failure.invalidLedger }
        try Self.controlFile(directory, name: "owner.lock", maximum: 0)
        var info = stat()
        guard fstatat(directory, "owner.lock", &info, AT_SYMLINK_NOFOLLOW) == 0,
              Self.fileIdentity(info) == record.lockIdentity,
              unlinkat(directory, "owner.lock", 0) == 0,
              unlinkat(parent, record.id, AT_REMOVEDIR) == 0 else { throw Failure.unsafeStorage }
    }
    private func reclaimOrphans(control: Int32, tracked: Set<String>) throws {
        let parent = try Self.child(control, "reservations", create: false); defer { Darwin.close(parent) }
        for id in try Self.names(parent, maximum: Self.recordLimit) where !tracked.contains(id) {
            let directory = try Self.child(parent, id, create: false); defer { Darwin.close(directory) }
            let entries = try Self.names(directory, maximum: 1)
            if entries.isEmpty {
                guard unlinkat(parent, id, AT_REMOVEDIR) == 0 else { throw Failure.unsafeStorage }
            } else {
                guard entries == ["owner.lock"] else { throw Failure.invalidLedger }
                // Orphans have no trusted allowance. A live untracked lease fails
                // admission closed instead of silently ignoring possible growth.
                let url = root.appendingPathComponent("storage-accounting/reservations/" + id)
                let probe = try WorkflowFileLease(directory: url, name: "owner.lock", mode: .shared, requireLocalAPFS: true)
                defer { probe.close() }
                guard probe.parentIdentity == (try Self.directoryIdentity(directory)) else { throw Failure.unsafeStorage }
                let record = Record(id: id, scopes: [], peakBytes: 0, directoryIdentity: probe.parentIdentity, lockIdentity: probe.identity)
                try reclaim(record: record, control: control)
            }
        }
        // Atomic ledger temporaries are private, bounded and never payload.
        for name in try Self.names(control, maximum: 8) where name.hasPrefix(".ledger-") {
            try Self.controlFile(control, name: name, maximum: Self.ledgerLimit)
            guard unlinkat(control, name, 0) == 0 else { throw Failure.unsafeStorage }
        }
    }
    private final class ScanBudget { var entries = 0 }
    private func snapshot(budget: ScanBudget) throws -> Snapshot {
        try transaction { _, ledger in try self.measure(ledger.reservations, budget: budget) }
    }
    private func measure(_ records: [Record], additional: (scopes: [Scope], peakBytes: Int64)? = nil, budget: ScanBudget = ScanBudget()) throws -> Snapshot {
        var charges = records.map { (scopes: $0.scopes, peakBytes: $0.peakBytes) }
        if let additional { charges.append(additional) }
        // Active scopes are excluded from the retained walk and scanned once separately.
        // This prevents growth between two scans from subtracting bytes never counted.
        let outside = try totalBytes(excluding: charges.flatMap(\.scopes), budget: budget)
        var retained = outside, growth: Int64 = 0
        for charge in charges {
            let actual = try bytes(in: charge.scopes, budget: budget)
            retained = try Self.add(retained, actual)
            growth = try Self.add(growth, max(0, charge.peakBytes - actual))
        }
        if let accountingRoot {
            for member in try accountingMembers(accountingRoot, budget: budget) where member.url.path != root.path {
                try beforeAccountingMemberOpen?(member.url)
                let peer = try WorkflowStorageAdmission(workspace: member.url, accountingRoot: nil, limits: limits, expectedIdentity: member.identity)
                let snapshot = try peer.snapshot(budget: budget)
                retained = try Self.add(retained, snapshot.retainedBytes)
                growth = try Self.add(growth, snapshot.reservedGrowthBytes)
            }
        }
        _ = try Self.add(retained, growth)
        return .init(retainedBytes: retained, reservedGrowthBytes: growth)
    }
    private func accountingMembers(_ application: URL, budget: ScanBudget) throws -> [(url: URL, identity: WorkflowFileIdentity)] {
        let fd = try Self.openDirectory(application); defer { Darwin.close(fd) }
        let applicationIdentity = try Self.directoryIdentity(fd)
        guard applicationIdentity == accountingIdentity else { throw Failure.unsafeStorage }
        var info = stat()
        guard fstatat(fd, "external-projects", &info, AT_SYMLINK_NOFOLLOW) == 0 else {
            if errno == ENOENT { return [(application, applicationIdentity)] }; throw Failure.unsafeStorage
        }
        let parent = try Self.child(fd, "external-projects", create: false)
        defer { Darwin.close(parent) }
        budget.entries += 1
        guard budget.entries <= limits.maximumEntries else { throw Failure.scanLimit }
        var result = [(url: application, identity: applicationIdentity)]
        for name in try Self.names(parent, maximum: min(limits.maximumEntries - budget.entries, 1000)) {
            guard UUID(uuidString: name)?.uuidString == name else { throw Failure.unsafeStorage }
            let child = try Self.child(parent, name, create: false)
            let identity = try Self.directoryIdentity(child)
            Darwin.close(child)
            budget.entries += 1
            guard budget.entries <= limits.maximumEntries else { throw Failure.scanLimit }
            result.append((application.appendingPathComponent("external-projects").appendingPathComponent(name), identity))
        }
        return result
    }
    private func bytes(in scopes: [Scope], budget: ScanBudget) throws -> Int64 {
        let fd = try Self.openDirectory(root); defer { Darwin.close(fd) }
        var total: Int64 = 0
        for scope in scopes { total = try Self.add(total, scanPath(fd, parts: scope.relativeComponents, count: &budget.entries)) }
        return total
    }
    private func totalBytes(excluding: [Scope], budget: ScanBudget) throws -> Int64 {
        let fd = try Self.openDirectory(root); defer { Darwin.close(fd) }
        var total: Int64 = 0
        for namespace in Namespace.allCases { total = try Self.add(total, scanPath(fd, parts: namespace.parts, count: &budget.entries, excluding: excluding)) }
        // Earlier preparation versions used a root-level private stage. Count only the exact
        // canonical UUID family, never arbitrary .local build/runtime/log directories.
        for name in try Self.names(fd, maximum: limits.maximumEntries) where name.hasPrefix("preparation-source-") {
            let id = String(name.dropFirst("preparation-source-".count))
            if UUID(uuidString: id)?.uuidString == id { total = try Self.add(total, scanPath(fd, parts: [name], count: &budget.entries, excluding: excluding)) }
        }
        return total
    }
    private func scanPath(_ root: Int32, parts: [String], count: inout Int, excluding: [Scope] = []) throws -> Int64 {
        var parent = dup(root)
        guard parent >= 0 else { throw Failure.unsafeStorage }
        defer { Darwin.close(parent) }
        for name in parts {
            var info = stat()
            if fstatat(parent, name, &info, AT_SYMLINK_NOFOLLOW) != 0 {
                if errno == ENOENT { return 0 }; throw Failure.unsafeStorage
            }
            count += 1; guard count <= limits.maximumEntries else { throw Failure.scanLimit }
            let next = try Self.child(parent, name, create: false)
            Darwin.close(parent); parent = next
        }
        return try walk(parent, parts: parts, count: &count, excluding: excluding)
    }
    private func walk(_ directory: Int32, parts: [String], count: inout Int, excluding: [Scope]) throws -> Int64 {
        if excluding.contains(where: { $0.relativeComponents == parts }) { return 0 }
        guard parts.count <= limits.maximumDepth else { throw Failure.scanLimit }
        var total: Int64 = 0
        for name in try Self.names(directory, maximum: limits.maximumEntries) {
            count += 1; guard count <= limits.maximumEntries else { throw Failure.scanLimit }
            var info = stat()
            guard fstatat(directory, name, &info, AT_SYMLINK_NOFOLLOW) == 0, info.st_uid == geteuid(), info.st_mode & 0o022 == 0 else { throw Failure.unsafeStorage }
            switch info.st_mode & mode_t(S_IFMT) {
            case mode_t(S_IFDIR):
                let child = try Self.child(directory, name, create: false)
                defer { Darwin.close(child) }
                guard try Self.directoryIdentity(child) == Self.fileIdentity(info) else { throw Failure.unsafeStorage }
                total = try Self.add(total, walk(child, parts: parts + [name], count: &count, excluding: excluding))
            case mode_t(S_IFREG):
                guard info.st_nlink == 1, info.st_size >= 0 else { throw Failure.unsafeStorage }
                total = try Self.add(total, Int64(info.st_size))
            default: throw Failure.unsafeStorage
            }
        }
        return total
    }
    /// Accounting records are excluded from payload totals, but their shape and size are
    /// bounded so this namespace cannot conceal arbitrary payload from admission.
    private func validateControl(_ control: Int32) throws {
        let names = try Self.names(control, maximum: 8)
        for name in names {
            if name == "reservations" {
                let parent = try Self.child(control, name, create: false); defer { Darwin.close(parent) }
                for id in try Self.names(parent, maximum: Self.recordLimit) {
                    guard UUID(uuidString: id)?.uuidString == id else { throw Failure.invalidLedger }
                    let child = try Self.child(parent, id, create: false); defer { Darwin.close(child) }
                    let entries = try Self.names(child, maximum: 1)
                    guard entries.isEmpty || entries == ["owner.lock"] else { throw Failure.invalidLedger }
                    if !entries.isEmpty { try Self.controlFile(child, name: "owner.lock", maximum: 0) }
                }
            } else if name == "owner.lock" || name == "activity.lock" { try Self.controlFile(control, name: name, maximum: 0) }
            else if name == "ledger.json" { try Self.controlFile(control, name: name, maximum: Self.ledgerLimit) }
            else if name.hasPrefix(".ledger-"), UUID(uuidString: String(name.dropFirst(8)))?.uuidString == String(name.dropFirst(8)) {
                try Self.controlFile(control, name: name, maximum: Self.ledgerLimit)
            } else { throw Failure.invalidLedger }
        }
    }
    private static func controlFile(_ directory: Int32, name: String, maximum: Int) throws {
        var info = stat()
        guard fstatat(directory, name, &info, AT_SYMLINK_NOFOLLOW) == 0,
              info.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG), info.st_nlink == 1,
              info.st_uid == geteuid(), info.st_mode & 0o022 == 0, info.st_size >= 0, info.st_size <= maximum else { throw Failure.invalidLedger }
    }
    private func load(_ control: Int32) throws -> Ledger {
        let file = openat(control, "ledger.json", O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        if file < 0 { if errno == ENOENT { return Ledger() }; throw Failure.invalidLedger }
        defer { Darwin.close(file) }
        var info = stat()
        guard fstat(file, &info) == 0, info.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG), info.st_nlink == 1,
              info.st_uid == geteuid(), info.st_mode & 0o022 == 0, info.st_size >= 0, info.st_size <= Self.ledgerLimit else { throw Failure.invalidLedger }
        var bytes = [UInt8](repeating: 0, count: Int(info.st_size)), offset = 0
        let length = bytes.count
        while offset < length {
            let result = bytes.withUnsafeMutableBytes { Darwin.read(file, $0.baseAddress!.advanced(by: offset), length - offset) }
            guard result > 0 else { throw Failure.invalidLedger }; offset += result
        }
        let ledger = try JSONDecoder().decode(Ledger.self, from: Data(bytes))
        guard ledger.schemaVersion == 1, ledger.reservations.count <= Self.recordLimit,
              Set(ledger.reservations.map(\.id)).count == ledger.reservations.count else { throw Failure.invalidLedger }
        for record in ledger.reservations {
            guard UUID(uuidString: record.id)?.uuidString == record.id, record.peakBytes > 0, !record.scopes.isEmpty, record.scopes.count <= 16 else { throw Failure.invalidLedger }
            for scope in record.scopes { try scope.validate() }
            if let authority = record.authority {
                try authority.validate()
                guard record.scopes.contains(authority.scope) else { throw Failure.invalidLedger }
            }
        }
        try Self.requireDisjoint(ledger.reservations.flatMap(\.scopes))
        return ledger
    }
    private func save(_ ledger: Ledger, control: Int32) throws {
        let bytes = try JSONEncoder().encode(ledger)
        guard bytes.count <= Self.ledgerLimit else { throw Failure.invalidLedger }
        let name = ".ledger-" + UUID().uuidString
        let fd = openat(control, name, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard fd >= 0 else { throw Failure.invalidLedger }
        defer { Darwin.close(fd); _ = unlinkat(control, name, 0) }
        try bytes.withUnsafeBytes { raw in
            var offset = 0
            while offset < raw.count {
                let written = Darwin.write(fd, raw.baseAddress!.advanced(by: offset), raw.count - offset)
                guard written > 0 else { throw Failure.invalidLedger }; offset += written
            }
        }
        guard fsync(fd) == 0, renameat(control, name, control, "ledger.json") == 0, fsync(control) == 0 else { throw Failure.invalidLedger }
    }
    private static func requireDisjoint(_ scopes: [Scope]) throws {
        for (index, scope) in scopes.enumerated() {
            if scopes.prefix(index).contains(where: { overlaps($0, scope) }) { throw Failure.overlappingScope }
        }
    }
    private static func overlaps(_ a: Scope, _ b: Scope) -> Bool {
        a.relativeComponents.starts(with: b.relativeComponents) || b.relativeComponents.starts(with: a.relativeComponents)
    }
    private static func add(_ a: Int64, _ b: Int64) throws -> Int64 {
        let (value, overflow) = a.addingReportingOverflow(b)
        guard !overflow, value >= 0 else { throw Failure.scanLimit }; return value
    }
    private static func fileIdentity(_ info: stat) -> WorkflowFileIdentity { .init(device: UInt64(UInt32(bitPattern: info.st_dev)), inode: UInt64(info.st_ino)) }
    private static func directoryIdentity(_ fd: Int32) throws -> WorkflowFileIdentity {
        var info = stat()
        guard fstat(fd, &info) == 0, info.st_mode & mode_t(S_IFMT) == mode_t(S_IFDIR), info.st_uid == geteuid(), info.st_mode & 0o022 == 0 else { throw Failure.unsafeStorage }
        return fileIdentity(info)
    }
    private static func openDirectory(_ url: URL) throws -> Int32 {
        let path = url.path
        guard url.isFileURL, path.hasPrefix("/"), !path.contains("\0"),
              url.host == nil || url.host == "" || url.host == "localhost",
              !path.split(separator: "/").contains(where: { $0 == "." || $0 == ".." }) else { throw Failure.unsafeStorage }
        // Open the authorized workspace directly: a sandbox need not allow reading
        // its ancestors. O_NOFOLLOW_ANY rejects symlinks throughout the path.
        let fd = open(path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW_ANY | O_CLOEXEC)
        guard fd >= 0 else { throw Failure.unsafeStorage }
        do { _ = try directoryIdentity(fd); return fd }
        catch { Darwin.close(fd); throw error }
    }

    private static func child(_ parent: Int32, _ name: String, create: Bool, exclusive: Bool = false) throws -> Int32 {
        if create, mkdirat(parent, name, 0o700) != 0, exclusive || errno != EEXIST { throw Failure.unsafeStorage }
        let fd = openat(parent, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard fd >= 0 else { throw Failure.unsafeStorage }
        do { _ = try directoryIdentity(fd); return fd } catch { Darwin.close(fd); throw error }
    }
    private static func names(_ directory: Int32, maximum: Int) throws -> [String] {
        let copy = dup(directory)
        guard copy >= 0 else { throw Failure.unsafeStorage }
        guard let stream = fdopendir(copy) else { Darwin.close(copy); throw Failure.unsafeStorage }
        defer { closedir(stream) }
        rewinddir(stream)
        var result: [String] = []
        errno = 0
        while let entry = readdir(stream) {
            let name = withUnsafePointer(to: &entry.pointee.d_name) { pointer in
                pointer.withMemoryRebound(to: CChar.self, capacity: Int(MAXNAMLEN) + 1) { String(cString: $0) }
            }
            if name != ".", name != ".." { result.append(name) }
            guard result.count <= maximum else { throw Failure.scanLimit }
            errno = 0
        }
        guard errno == 0 else { throw Failure.unsafeStorage }
        return result
    }
}
