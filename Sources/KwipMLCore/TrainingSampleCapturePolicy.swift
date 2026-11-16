import Foundation

/// Optional observations. Historical v1 policies keep performance capture off.
public struct TrainingSampleCapturePolicy: Codable, Equatable {
    public static let heatmapFrameBounds = 0...24
    public let schemaVersion: Int
    public let heatmapFrameLimit:Int
    public let tensorHeatmaps:Bool
    public let activationValues: Bool
    public let predictionRows: Bool
    public let performanceMeasurements: Bool
    public let retainComponentOutputs:Bool?
    public let retainDiagnosticDetails:Bool?
    public var savesComponentOutputs:Bool {retainComponentOutputs ?? true}
    public var savesDiagnosticDetails:Bool {retainDiagnosticDetails ?? true}
    public func applyingHistory(outputs:Bool,diagnostics:Bool,requiresPredictionTable:Bool)->Self {
        .init(activationValues:activationValues && outputs,predictionRows:(predictionRows && outputs) || requiresPredictionTable,performanceMeasurements:performanceMeasurements && diagnostics,retainComponentOutputs:outputs,retainDiagnosticDetails:diagnostics,tensorHeatmaps:schemaVersion>=4 ? tensorHeatmaps && outputs:nil,heatmapFrameLimit:outputs ? heatmapFrameLimit:0)
    }
    public static let factory = Self(activationValues: true, predictionRows: true)

    public init(activationValues: Bool, predictionRows: Bool, performanceMeasurements: Bool? = nil,retainComponentOutputs:Bool?=nil,retainDiagnosticDetails:Bool?=nil,tensorHeatmaps:Bool?=nil,heatmapFrameLimit:Int=0) {
        schemaVersion = tensorHeatmaps != nil ? 4:retainComponentOutputs != nil || retainDiagnosticDetails != nil ? 3:performanceMeasurements == nil ? 1 : 2
        self.tensorHeatmaps=tensorHeatmaps ?? false
        self.heatmapFrameLimit=heatmapFrameLimit
        self.retainComponentOutputs=schemaVersion>=3 ? retainComponentOutputs ?? true:nil
        self.retainDiagnosticDetails=schemaVersion>=3 ? retainDiagnosticDetails ?? true:nil
        self.activationValues = activationValues
        self.predictionRows = predictionRows
        self.performanceMeasurements = performanceMeasurements ?? false
    }
    public func changing(activationValues:Bool?=nil,predictionRows:Bool?=nil,performanceMeasurements:Bool?=nil,tensorHeatmaps:Bool?=nil,heatmapFrameLimit:Int?=nil)->Self {
        .init(activationValues:activationValues ?? self.activationValues,predictionRows:predictionRows ?? self.predictionRows,
              performanceMeasurements:performanceMeasurements ?? (schemaVersion>=2 ? self.performanceMeasurements:nil),
              retainComponentOutputs:retainComponentOutputs,retainDiagnosticDetails:retainDiagnosticDetails,
              tensorHeatmaps:tensorHeatmaps ?? (schemaVersion>=4 ? self.tensorHeatmaps:nil),heatmapFrameLimit:(tensorHeatmaps == false) ? 0:heatmapFrameLimit ?? self.heatmapFrameLimit)
    }
    private enum CodingKeys: String, CodingKey { case schemaVersion, heatmapFrameLimit, tensorHeatmaps, activationValues, predictionRows, performanceMeasurements, retainComponentOutputs, retainDiagnosticDetails }
    private struct Field: CodingKey {
        let stringValue: String
        var intValue: Int? { nil }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { return nil }
    }
    public init(from decoder: Decoder) throws {
        let fields = Set(try decoder.container(keyedBy: Field.self).allKeys.map(\.stringValue))
        let values = try decoder.container(keyedBy: CodingKeys.self)
        let version = try values.decode(Int.self, forKey: .schemaVersion)
        let expected: Set<String> = ["schemaVersion", "activationValues", "predictionRows"]
        guard (1...4).contains(version), fields == (version >= 3 ? expected.union(["performanceMeasurements","retainComponentOutputs","retainDiagnosticDetails"]).union(version==4 ? ["tensorHeatmaps","heatmapFrameLimit"]:[]):version == 2 ? expected.union(["performanceMeasurements"]) : expected) else {
            throw GraphConfigurationError("Unknown or missing training sample capture fields or unsupported schema.")
        }
        self.init(activationValues: try values.decode(Bool.self, forKey: .activationValues),
                  predictionRows: try values.decode(Bool.self, forKey: .predictionRows),
                  performanceMeasurements: version >= 2 ? try values.decode(Bool.self, forKey: .performanceMeasurements) : nil,retainComponentOutputs:version>=3 ? try values.decode(Bool.self,forKey:.retainComponentOutputs):nil,retainDiagnosticDetails:version>=3 ? try values.decode(Bool.self,forKey:.retainDiagnosticDetails):nil,tensorHeatmaps:version==4 ? try values.decode(Bool.self,forKey:.tensorHeatmaps):nil,heatmapFrameLimit:version==4 ? try values.decode(Int.self,forKey:.heatmapFrameLimit):0)
        guard Self.heatmapFrameBounds.contains(heatmapFrameLimit),heatmapFrameLimit==0 || tensorHeatmaps, savesComponentOutputs || (!activationValues && !tensorHeatmaps),savesDiagnosticDetails || !performanceMeasurements else {throw GraphConfigurationError("Disabled History retention cannot collect optional observations.")}
    }
    public func encode(to encoder: Encoder) throws {
        guard Self.heatmapFrameBounds.contains(heatmapFrameLimit),heatmapFrameLimit==0 || tensorHeatmaps, savesComponentOutputs || (!activationValues && !tensorHeatmaps),savesDiagnosticDetails || !performanceMeasurements else {throw GraphConfigurationError("Disabled History retention cannot collect optional observations.")}
        var fields = encoder.container(keyedBy: CodingKeys.self)
        try fields.encode(schemaVersion, forKey: .schemaVersion)
        if schemaVersion==4 {try fields.encode(tensorHeatmaps,forKey:.tensorHeatmaps);try fields.encode(heatmapFrameLimit,forKey:.heatmapFrameLimit)}
        try fields.encode(activationValues, forKey: .activationValues)
        try fields.encode(predictionRows, forKey: .predictionRows)
        if schemaVersion >= 2 { try fields.encode(performanceMeasurements, forKey: .performanceMeasurements) }
        if schemaVersion>=3 {try fields.encode(retainComponentOutputs,forKey:.retainComponentOutputs);try fields.encode(retainDiagnosticDetails,forKey:.retainDiagnosticDetails)}
    }
}
