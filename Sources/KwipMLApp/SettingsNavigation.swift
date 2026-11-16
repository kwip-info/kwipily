import KwipMLCore
import Foundation
import Combine

/// Shared window navigation; explicit links reveal their page without rebuilding draft editors.
@MainActor final class AppSettingsNavigation: ObservableObject {
    @Published var selected: SettingsPage = .appearance
    @Published var search = ""
    func open(_ page: SettingsPage) {
        search = ""
        selected = page
    }
}

enum SettingsPage: String, CaseIterable, Identifiable {
    case models, mcp, updates, support, appearance, canvas, shortcuts, feedback, learning, execution, schedules, webhooks, storage, permissions
    var id: String { rawValue }
    var title: String {
        switch self {
        case .webhooks:return "Webhooks"
        case .schedules:return "Scheduled runs"
        case .models: return "Models & assets"
        case .mcp: return "MCP"
        case .support: return "Support"
        case .updates: return "Updates"
        case .appearance: return "Appearance"
        case .canvas: return "Canvas & workspace"
        case .shortcuts: return "Keyboard shortcuts"
        case .feedback: return "Run feedback"
        case .learning: return "Learning progress"
        case .execution: return "Execution & capture"
        case .storage: return "Storage & recovery"
        case .permissions: return "Permissions"
        }
    }
    var group: String {
        switch self { case .updates, .appearance, .canvas, .shortcuts, .feedback, .learning: return "Everyday"
        case .webhooks, .schedules, .models, .execution, .storage: return "Resources"
        case .mcp, .permissions, .support: return "Access" }
    }
    var symbol: String {
        switch self { case .webhooks:return "link";case .schedules:return "calendar";case .models:return "shippingbox"; case .updates: return "arrow.down.circle"; case .appearance: return "paintpalette"; case .canvas: return "square.grid.2x2"
        case .mcp: return "point.3.connected.trianglepath.dotted"
        case .support: return "questionmark.circle"
        case .shortcuts: return "keyboard"; case .feedback: return "bell"
        case .learning: return "graduationcap"
        case .execution: return "slider.horizontal.3"; case .storage: return "externaldrive"
        case .permissions: return "lock" }
    }
    var subtitle: String {
        switch self {
        case .webhooks:return "Receive authenticated local requests and run saved workflows."
        case .schedules:return "Schedule saved workflows and inspect local job history."
        case .models: return "Review local models, captured media, Hub access and protected temporary cleanup."
        case .mcp: return "Connect an AI client to selected local workflows."
        case .support: return "Get help and review a private support report."
        case .updates: return "Check for new v1 releases."
        case .appearance: return "Make \(ProductBrand.name) feel at home on your Mac."
        case .canvas: return "Canvas presentation and window recovery on this Mac."
        case .shortcuts: return "Make everyday actions easy to reach."
        case .feedback: return "Choose how \(ProductBrand.name) keeps you informed."
        case .learning: return "Review and reset saved checks without deleting your work."
        case .execution: return "Resource and sample-capture defaults for future runs."
        case .storage: return "Allowances and recovery for local work on this Mac."
        case .permissions: return "Manage saved access to your local sources."
        }
    }
    private var keywords: String {
        switch self {
        case .webhooks:return "webhook inbound trigger event delivery local port authentication signature bearer JSON payload mapping"
        case .schedules:return "schedule scheduled runs jobs history timer interval daily weekly project saved versions automation"
        case .models: return "models downloads hugging face hub credentials tokens limits cleanup assets offline allowlist network"
        case .mcp: return "mcp ai agent automation client port permissions tools projects"
        case .support: return "support help contact diagnostics report privacy export error failure"
        case .updates: return "update updates download install version automatic check sparkle"
        case .appearance: return "color mode light dark system rose teal accent transparency panels"
        case .canvas: return "left right rounded table nodes workspace layout launch reopen windows scroll wheel trackpad zoom sensitivity"
        case .shortcuts: return "keys keybindings keyboard command shortcut"
        case .feedback: return "notification notifications notify completion failure status icon menu bar"
        case .learning: return "learning progress reset lessons attempts checks completion"
        case .execution: return "CPU GPU MPS threads time allowance worker memory RSS activation prediction sample capture runtime refresh status verified devices PyTorch"
        case .storage: return "cache preparation managed storage bytes usage recovery restore retained records authoring evidence"
        case .permissions: return "source access file revoke permissions privacy"
        }
    }
    static func matches(_ query: String) -> [Self] {
        let words = query.lowercased().split(whereSeparator: { $0.isWhitespace })
        return allCases.filter { page in
            let searchable = (page.title + " " + page.keywords).lowercased()
            return words.allSatisfy { searchable.contains($0) }
        }
    }
}
