import SwiftUI
import KwipMLCore

/// Identity follows registry categories, never an inferred ML operation or port payload.
/// One pair is shared by generated glyphs, symbol fallbacks, frames and canvas accents.
enum ComponentVisualFamily: String, CaseIterable {
    case data, layers, activations, merge, tensorLayout, normalization, boundaries
    case embeddings, tensorMath, pooling, training, evaluation, blocks, unknown

    private static let registered: [String: String] = {
        guard let registry = try? ComponentRegistry.builtIn() else { return [:] }
        return Dictionary(uniqueKeysWithValues: registry.components.map { ($0.id, $0.category) })
    }()
    static func resolve(type: String) -> Self {
        if type.hasPrefix("kwip.data.") { return .data }
        if type=="parallel_features" {return .merge}
        if WorkflowModelInput.componentIDs.contains(type) || type=="workflow.modelDefinition" || type=="output" {return .boundaries}
        if type=="workflow.frozenModel" {return .boundaries}
        if type=="workflow.saveModel" {return .training}
        if type == ActivationPalette.type {return .activations}
        if type.hasPrefix("kwip.tensor.") {return .tensorLayout}
        if type == "kwip.reshape" {return .tensorLayout}
        if type == "kwip.linear" || type == "kwip.recurrent" || type == "kwip.recurrentCell" {return .layers}
        if type == "dropout" || type == "kwip.normalization" {return .normalization}
        if type == "kwip.convolution" || type == "separable_conv2d" {return .layers}
        if type == PoolingPalette.type {return .pooling}
        if type.hasPrefix("composite:") || type.hasPrefix("kwip.composition.") || type == "kwip.block" { return .blocks }
        switch type {
        case "dataset_source", "data_loader", "train", "workflow.modelDefinition", "workflow.trainingData", "workflow.runSetup", "workflow.train", "workflow.logisticRegression", "workflow.kerasClassifier": return .training
        case "workflow.gameSession", "workflow.generateText", "workflow.frozenModel", "workflow.evaluateTest", "objective_check": return .evaluation
        default: return category(registered[type] ?? BundledPoolingComponents.entries[type]?.definition.category ?? "")
        }
    }
    static func category(_ name: String) -> Self {
        switch name {
        case "Data": return .data
        case "Layers", "Attention": return .layers
        case "Activations": return .activations
        case "Merge": return .merge
        case "Tensor layout": return .tensorLayout
        case "Normalization", "Conditioning": return .normalization
        case "Boundaries": return .boundaries
        case "Embeddings": return .embeddings
        case "Tensor math": return .tensorMath
        case "Pooling": return .pooling
        case "Training": return .training
        case "Evaluation": return .evaluation
        case "Blocks", "Architecture": return .blocks
        default: return .unknown
        }
    }
    var colors: (primary: UInt32, secondary: UInt32) {
        switch self {
        case .data: return (0x79B8B0, 0x387D77)
        case .layers: return (0xC9849D, 0x8C4863)
        case .activations: return (0xD7B57B, 0x947443)
        case .merge: return (0x8CAECB, 0x4A6C8E)
        case .tensorLayout: return (0xB3A3D4, 0x76638F)
        case .normalization: return (0xB8BB8E, 0x757945)
        case .boundaries: return (0xBAADA0, 0x756655)
        case .embeddings: return (0xB49ED1, 0x765791)
        case .tensorMath: return (0x99B8C8, 0x527588)
        case .pooling: return (0xB8BB8E, 0x757945)
        case .training: return (0xA7BA8F, 0x62764D)
        case .evaluation: return (0x90BABE, 0x477A80)
        case .blocks: return (0xC99FB7, 0x80536E)
        case .unknown: return (0xACA8B2, 0x6D6876)
        }
    }
    var primary: Color { Self.color(colors.primary) }
    var secondary: Color { Self.color(colors.secondary) }
    var silhouette: ComponentIconSilhouette {
        switch self {
        case .data, .boundaries: return .square
        case .layers, .embeddings: return .hexagon
        case .activations: return .circle
        case .merge, .tensorMath: return .diamond
        case .tensorLayout: return .rectangle
        case .normalization, .pooling: return .octagon
        case .training, .evaluation, .blocks: return .clippedSquare
        case .unknown: return .square
        }
    }
    private static func color(_ value: UInt32) -> Color {
        Color(red: Double((value >> 16) & 255) / 255, green: Double((value >> 8) & 255) / 255, blue: Double(value & 255) / 255)
    }
}

enum ComponentIconSilhouette: String, CaseIterable { case square, hexagon, circle, diamond, rectangle, octagon, clippedSquare }

struct ComponentIconFrame: Shape {
    let silhouette: ComponentIconSilhouette
    func path(in rect: CGRect) -> Path {
        switch silhouette {
        case .square: return Path(rect)
        case .circle: return Path(ellipseIn: rect)
        case .rectangle: return Path(CGRect(x: rect.minX, y: rect.minY + rect.height * 0.1, width: rect.width, height: rect.height * 0.8))
        default:
            let points: [(CGFloat, CGFloat)]
            switch silhouette {
            case .hexagon: points = [(0.25,0),(0.75,0),(1,0.5),(0.75,1),(0.25,1),(0,0.5)]
            case .diamond: points = [(0.5,0),(1,0.5),(0.5,1),(0,0.5)]
            default:
                let cut: CGFloat = silhouette == .octagon ? 0.27 : 0.12
                points = [(cut,0),(1-cut,0),(1,cut),(1,1-cut),(1-cut,1),(cut,1),(0,1-cut),(0,cut)]
            }
            var result = Path()
            result.addLines(points.map { CGPoint(x: rect.minX + $0.0 * rect.width, y: rect.minY + $0.1 * rect.height) })
            result.closeSubpath()
            return result
        }
    }
}
