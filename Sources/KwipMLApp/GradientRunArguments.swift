import Foundation
import KwipMLCore

enum GradientRunArguments {
    static func arguments(config: GradientCaptureConfiguration?, supportedDevices: [String], device: String) throws -> [String] {
        guard let config else { return [] }
        try config.validate()
        guard !supportedDevices.isEmpty, device == "auto" || supportedDevices.contains(device) else {
            throw NSError(domain: "KwipML.GradientCapture", code: 1, userInfo: [NSLocalizedDescriptionKey: "This runtime has not verified gradient capture for the selected device. Turn capture off or select a supported runtime and device in Project settings."])
        }
        return ["--capture-gradients", "--gradient-every-epochs", String(config.everyEpochs), "--gradient-max-batches-per-epoch", String(config.maxBatchesPerEpoch), "--gradient-max-captures", String(config.maxCaptures)]
    }
}
