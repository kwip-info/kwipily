import Foundation

/// Project-scoped routes only. Global appearance, permissions and storage allowances
/// stay in app Settings; universal training configuration stays on its components.
enum ProjectConfigurePage: String, CaseIterable, Identifiable {
    case identity, files, schedules, webhooks, training, resources, capture, workspace
    var id: String { rawValue }
    var group: String { [.identity, .files, .workspace].contains(self) ? "Project" : "Execution" }
    var title: String {
        switch self {
        case .webhooks:return "Webhooks"
        case .schedules:return "Scheduled runs"
        case .identity: return "Identity"
        case .files: return "Project files"
        case .training: return "Training"
        case .resources: return "Resource limits"
        case .capture: return "Output & capture"
        case .workspace: return "Workspace"
        }
    }
    var symbol: String {
        switch self {
        case .webhooks:return "link"
        case .schedules:return "calendar"
        case .identity: return "doc.text"
        case .files: return "folder"
        case .training: return "point.3.connected.trianglepath.dotted"
        case .resources: return "slider.horizontal.3"
        case .capture: return "waveform.path"
        case .workspace: return "rectangle.3.group"
        }
    }
    func subtitle(universal: Bool) -> String {
        switch self {
        case .webhooks:return "Receive authenticated local events for a saved version of this project."
        case .schedules:return "Schedule saved versions of this workflow and inspect its jobs."
        case .identity: return "Name and identify this project."
        case .files: return "Save, export and locate the current project."
        case .training: return universal ? "Configure each training path on its connected components." : "Control iteration, reproducibility and the optimizer for this project."
        case .resources: return "Project overrides for future operations on this Mac."
        case .capture: return "Choose the observations retained by future runs."
        case .workspace: return "Control this canvas’s overview and project panels."
        }
    }
    private var keywords: String {
        switch self {
        case .webhooks:return "webhook inbound trigger event delivery authentication JSON table mapping port"
        case .schedules:return "schedule scheduled runs jobs history timer interval daily weekly saved version automation"
        case .identity: return "name revision components connections definitions"
        case .files: return "save export reveal portable local files"
        case .training: return "run setup training data train device epochs batch sample seed split validation optimizer learning rate"
        case .resources: return "CPU threads time memory RSS inheritance overrides cache storage usage"
        case .capture: return "output activation prediction sample capture gradient"
        case .workspace: return "workspace canvas left right rounded table nodes ports minimap overview ribbon configuration results panel visible hidden"
        }
    }
    static func pages(universal: Bool) -> [Self] {
        allCases.filter { universal ? $0 != .training : ![.files,.schedules,.webhooks].contains($0) }
    }
    static func matches(_ query: String, universal: Bool) -> [Self] {
        let words = query.lowercased().split(whereSeparator: { $0.isWhitespace })
        return pages(universal: universal).filter { page in
            let text = (page.title + " " + page.keywords).lowercased()
            return words.allSatisfy { text.contains($0) }
        }
    }
}
