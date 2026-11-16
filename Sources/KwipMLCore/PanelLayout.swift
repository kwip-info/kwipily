import Foundation

/// Stable semantic layout identities; conditional panes never borrow a differently shaped layout.
public enum WorkspacePanelScope: String, Codable, CaseIterable {
    case modelLibraryInspector = "model.outer.library-inspector"
    case modelLibrary = "model.outer.library"
    case modelInspector = "model.outer.inspector"
    case modelCanvas = "model.outer.canvas"
    case modelPreview = "model.preview"
    case modelRibbonInspector = "model.ribbon.inspector"
    case dataRibbonInspector = "data.ribbon.inspector"
    case compositeRibbonInspector = "composite.ribbon.inspector"
    case affineRibbonDock = "affine.ribbon.dock"
    case dataExpanded = "data.outer.expanded"
    case dataCompact = "data.outer.compact"
    case dataHidden = "data.outer.hidden"
    case dataPreparation = "data.preparation"
    /// New docks require explicit initial sizing instead of AppKit intrinsic maximum width.
    public var initialPreference:PanelLayoutPreference? {
        guard [.modelRibbonInspector,.dataRibbonInspector,.compositeRibbonInspector,.affineRibbonDock].contains(self) else {return nil}
        return PanelLayoutPreference(sizes:Dictionary(uniqueKeysWithValues:panes.map {($0.id,$0.ideal)}))
    }
    public var vertical: Bool { self != .modelPreview }
    public var panes: [PanelPane] {
        let canvas = PanelPane("canvas", minimum: 500, ideal: 800, maximum: 100_000, flexible: true)
        let library = PanelPane("library", minimum: 175, ideal: 205, maximum: 270)
        let inspector = PanelPane("inspector", minimum: 225, ideal: 255, maximum: 330)
        switch self {
        case .modelLibraryInspector: return [library, canvas, inspector]
        case .modelLibrary: return [library, canvas]
        case .modelInspector: return [canvas, inspector]
        case .modelRibbonInspector: return [PanelPane("inspector", minimum: 225, ideal: 280, maximum: 380), canvas]
        case .dataRibbonInspector: return [PanelPane("inspector", minimum: 250, ideal: 290, maximum: 360), PanelPane("canvas", minimum: 380, ideal: 600, maximum: 100_000, flexible: true)]
        case .compositeRibbonInspector: return [PanelPane("inspector", minimum: 240, ideal: 300, maximum: 420), PanelPane("canvas", minimum: 350, ideal: 600, maximum: 100_000, flexible: true)]
        case .affineRibbonDock: return [PanelPane("dock", minimum: 300, ideal: 360, maximum: 520), PanelPane("canvas", minimum: 350, ideal: 600, maximum: 100_000, flexible: true)]
        case .modelCanvas, .dataHidden: return [canvas]
        case .modelPreview: return [PanelPane("canvas", minimum: 260, ideal: 500, maximum: 100_000, flexible: true), PanelPane("preview", minimum: 100, ideal: 170, maximum: 300)]
        case .dataExpanded: return [PanelPane("navigation", minimum: 175, ideal: 205, maximum: 270), canvas]
        case .dataCompact: return [PanelPane("navigation", minimum: 54, ideal: 54, maximum: 54), canvas]
        case .dataPreparation: return [PanelPane("palette", minimum: 160, ideal: 190, maximum: 240), PanelPane("canvas", minimum: 380, ideal: 600, maximum: 100_000, flexible: true), PanelPane("inspector", minimum: 250, ideal: 290, maximum: 360)]
        }
    }
}
public struct PanelPane: Equatable {
    public let id: String
    public let minimum: Double
    public let ideal: Double
    public let maximum: Double
    public let flexible: Bool
    public init(_ id: String, minimum: Double, ideal: Double, maximum: Double, flexible: Bool = false) {
        self.id = id; self.minimum = minimum; self.ideal = ideal; self.maximum = maximum; self.flexible = flexible
    }
}
public struct PanelLayoutPreference: Codable, Equatable {
    public var sizes: [String: Double]
    public init(sizes: [String: Double]) { self.sizes = sizes }
    public func validate(scope: WorkspacePanelScope) throws {
        guard Set(sizes.keys) == Set(scope.panes.map(\.id)), sizes.values.allSatisfy({ $0.isFinite && (1...100_000).contains($0) }) else {
            throw WorkspaceArrangementError("Invalid panel size or semantic pane identity.")
        }
    }
}
public enum WorkspacePanelSizing {
    /// Copy only matching semantic pane identities; retain old scopes for historical arrangements.
    /// Explicit ribbon intent wins. Inspector-only legacy model scope precedes its library variant.
    public static func ribbonPreferences(_ saved:[String:PanelLayoutPreference])->[String:PanelLayoutPreference] {
        var result=saved
        let migrations:[(WorkspacePanelScope,[WorkspacePanelScope])]=[
            (.modelRibbonInspector,[.modelInspector,.modelLibraryInspector]),
            (.dataRibbonInspector,[.dataPreparation])]
        for (target,sources) in migrations where result[target.rawValue] == nil {
            for source in sources {
                guard let old=saved[source.rawValue],(try? old.validate(scope:source)) != nil else {continue}
                let sizes=Dictionary(uniqueKeysWithValues:target.panes.map {($0.id,old.sizes[$0.id] ?? $0.ideal)})
                result[target.rawValue]=PanelLayoutPreference(sizes:sizes)
                break
            }
        }
        return result
    }

    /// Adjacent pane gaps work for flipped and non-flipped native coordinate systems.
    public static func isDividerHit(coordinate:Double,paneRanges:[ClosedRange<Double>]) -> Bool {
        guard coordinate.isFinite,paneRanges.count>1 else{return false}
        return zip(paneRanges,paneRanges.dropFirst()).contains { first,second in
            let low=min(first.upperBound,second.upperBound)
            let high=max(first.lowerBound,second.lowerBound)
            return low<=high && coordinate>=low-2 && coordinate<=high+2
        }
    }

    /// Available excludes divider thickness. Preferences are input-only; temporary clamping cannot mutate intent.
    public static func sizes(scope: WorkspacePanelScope, available: Double, preference: PanelLayoutPreference?) -> [Double] {
        guard available.isFinite, available > 0, available <= 100_000 else { return [] }
        let panes = scope.panes
        let minimum = panes.map(\.minimum)
        let totalMinimum = minimum.reduce(0, +)
        if available < totalMinimum { return minimum.map { $0 / totalMinimum * available } }
        let valid = preference.flatMap { (try? $0.validate(scope: scope)) == nil ? nil : $0 }
        var sizes = panes.map { min($0.maximum, max($0.minimum, valid?.sizes[$0.id] ?? $0.ideal)) }
        let flex = panes.firstIndex(where: \.flexible) ?? 0
        sizes[flex] = panes[flex].minimum
        var remainder = available - sizes.reduce(0, +)
        if remainder >= 0 { sizes[flex] += remainder; return sizes }
        // Reduce fixed panes toward their minima only as much as the window requires.
        let slack = sizes.enumerated().map { $0.element - minimum[$0.offset] }.reduce(0, +)
        if slack > 0 {
            for index in sizes.indices { sizes[index] += remainder * ((sizes[index] - minimum[index]) / slack) }
        }
        remainder = available - sizes.reduce(0, +)
        sizes[flex] += remainder
        return sizes
    }
}
