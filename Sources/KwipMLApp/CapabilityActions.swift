import Foundation
import KwipMLCore

extension WorkspaceStore {
  var runtimeContractValidation: ContractValidation? {
    guard let version = runtimeContractVersion else { return nil }
    return BuiltInCapabilityBridge.validate(
      graph: graph, registry: registry, runtimeVersion: version)
  }
  var runtimeContractDeferredIssues: [ContractIssue] {
    runtimeContractValidation?.issues.filter { $0.severity == .deferred } ?? []
  }
  var verifiedRuntimeComponentCount: Int {
    guard runtimeExecutionContract != nil, runtimeContractVersion != nil else { return 0 }
    return registry.components.filter { ($0.id != "mixture_experts" || mixtureExpertsRuntimeAvailable) && (!EmbeddingBagRuntimeCapabilities.operations.contains($0.id) || embeddingBagRuntimeCapabilities.isComplete) && (BundledGradientControlComponents.metadata($0.id) == nil || gradientControlRuntimeCapabilities.contains($0.id)) && (!["dataset_source","data_loader"].contains($0.id) || dataLoadingRuntimeCapabilities.contains($0.id)) && ($0.id != "initialization" || initializationRuntimeCapabilities.contains($0.id)) && (!["index_input","embedding"].contains($0.id) || embeddingRuntimeCapability != nil) && ($0.id != "rms_norm" || rmsNormRuntimeCapability != nil) && ($0.id != "layer_norm" || layerNormRuntimeCapability != nil) && ($0.id != "matmul" || matrixMultiplyRuntimeCapability != nil) && ($0.id != "conv1d" || conv1dRuntimeCapability != nil) && ($0.id != "conv2d" || conv2dRuntimeCapability != nil) && (BundledPoolingComponents.entries[$0.id] == nil || poolingRuntimeCapabilities.contains($0.id)) && (BundledTensorCompositionComponents.entries[$0.id] == nil || tensorCompositionRuntimeCapabilities.contains($0.id)) && (!RemainingActivationRuntimeCapabilities.operations.contains($0.id) || remainingActivationRuntimeCapabilities.contains($0.id)) && (!StatefulNormalizationRuntimeCapabilities.operations.contains($0.id) || statefulNormalizationRuntimeCapabilities.contains($0.id)) && (!OptimizerConfiguration.componentIDs.contains($0.id) || optimizerRuntimeCapabilities.contains($0.id)) && (!SchedulerConfiguration.componentIDs.contains($0.id) || schedulerRuntimeCapabilities.contains($0.id)) && (!LossConfiguration.componentIDs.contains($0.id) || lossRuntimeCapabilities.contains($0.id)) && (!RecurrentConfiguration.componentIDs.contains($0.id) || recurrentRuntimeCapabilities.contains($0.id)) && (!AttentionConfiguration.componentIDs.contains($0.id) || attentionRuntimeCapabilities.contains($0.id)) && (!SpatialPatchRuntimeCapabilities.operations.contains($0.id) || spatialPatchRuntimeCapabilities.contains($0.id)) && (!TensorRoutingRuntimeCapabilities.operations.contains($0.id) || tensorRoutingRuntimeCapabilities.contains($0.id)) && (!ConfigurableActivationRuntimeCapabilities.operations.contains($0.id) || configurableActivationRuntimeCapabilities.contains($0.id)) && ($0.id != "depthwise_conv2d" || depthwiseConv2dRuntimeCapability != nil) && ($0.id != "conv_transpose3d" || convTranspose3dRuntimeCapability != nil) && ($0.id != "conv_transpose2d" || convTranspose2dRuntimeCapability != nil) && ($0.id != "conv_transpose1d" || convTranspose1dRuntimeCapability != nil) && ($0.id != "conv3d" || conv3dRuntimeCapability != nil) && ($0.id != "lazy_linear" || lazyLinearRuntimeCapability != nil) && ($0.id != "bilinear" || bilinearRuntimeCapability != nil) && ($0.id != "softmax" || softmaxRuntimeCapability != nil) && ($0.id != "gelu" || geluRuntimeCapability != nil) && ($0.id != "sigmoid" || sigmoidRuntimeCapability != nil) && (!TensorLayoutRuntimeCapability.operations.contains($0.id) || tensorLayoutRuntimeCapability != nil) }.count
  }
  var capabilityBoundaryError: String? { capabilityBoundaryError(for: graph) }
  func capabilityBoundaryError(for graph: GraphDocument, context: GraphValidationContext = .bundledDataset) -> String? {
    guard runtimeExecutionContract != nil else {
      return
        "The worker execution contract has not been verified. Check runtime status in Settings."
    }
    guard let version = runtimeContractVersion else {
      return "The exact PyTorch runtime has not been verified. Check runtime status in Settings."
    }
    let validation = BuiltInCapabilityBridge.validate(graph: graph, registry: registry, runtimeVersion: version, context: context)
    if graph.nodes.contains(where:{EmbeddingBagRuntimeCapabilities.operations.contains($0.type)}), !embeddingBagRuntimeCapabilities.isComplete {
      return "Bag Input and EmbeddingBag require verified paired runtime support for integer coordinates, whole groups and floating source weights. Refresh runtime status or select a compatible runtime."
    }
    if graph.nodes.contains(where: { ["index_input","embedding"].contains($0.type) }), embeddingRuntimeCapability == nil {
      return "Integer inputs and embeddings require verified runtime support for lookup types, optimizer restrictions and cached weight changes. Refresh runtime status or select a compatible runtime."
    }
    if graph.nodes.contains(where: {$0.type == "rms_norm"}),rmsNormRuntimeCapability == nil {
      return "RMS normalization requires verified scale-only runtime support. Refresh runtime status or select a compatible runtime."
    }
    if graph.nodes.contains(where: { $0.type == "layer_norm" }), layerNormRuntimeCapability == nil {
      return "Layer normalization requires verified runtime support for trailing sample dimensions, epsilon and affine parameters. Refresh runtime status or select a compatible runtime."
    }
    if let node=graph.nodes.first(where:{AttentionConfiguration.componentIDs.contains($0.type) && !attentionRuntimeCapabilities.contains($0.type)}) {
      return "\(node.name) requires verified runtime support for attention sources, masks, named outputs and learned tables. Refresh runtime status or select a compatible runtime."
    }
    if let node=graph.nodes.first(where:{RecurrentConfiguration.componentIDs.contains($0.type) && !recurrentRuntimeCapabilities.contains($0.type)}) {
      return "\(node.name) requires verified runtime support for explicit sequence/state shapes and learned memory parameters. Refresh runtime status or select a compatible runtime."
    }
    let requestedDevice=graph.execution?.device ?? execution.device
    let selectedDevice=requestedDevice == "auto" ? (availableDevices.contains("mps") ? "mps":"cpu"):requestedDevice
    if let node=graph.nodes.first(where:{RecurrentConfiguration.componentIDs.contains($0.type) && (try? RecurrentConfiguration(componentID:$0.type,values:$0.config).supports(device:selectedDevice)) == false}) {
      return "\(node.name) uses projected LSTM, which this runtime supports on CPU. Choose CPU in Train’s Run settings or set projected hidden width to zero."
    }
    if let node = graph.nodes.first(where: { StatefulNormalizationRuntimeCapabilities.operations.contains($0.type) && !statefulNormalizationRuntimeCapabilities.contains($0.type) }) {
      let title = BundledNormalizationComponents.entries[node.type]?.definition.title ?? node.type
      return "\(title) requires verified runtime support for channel groups, learned adjustments and saved statistics. Refresh runtime status or select a compatible runtime."
    }
    if let node = graph.nodes.first(where: { SpatialPatchRuntimeCapabilities.operations.contains($0.type) && !spatialPatchRuntimeCapabilities.contains($0.type) }) {
      let title = BundledSpatialPatchComponents.entries[node.type]?.definition.title ?? node.type
      return "\(title) requires verified runtime support for patch geometry, zero borders and overlapping addition. Refresh runtime status or select a compatible runtime."
    }
    if let node = graph.nodes.first(where: { TensorRoutingRuntimeCapabilities.operations.contains($0.type) && !tensorRoutingRuntimeCapabilities.contains($0.type) }) {
      let title = BundledTensorRoutingComponents.entries[node.type]?.definition.title ?? node.type
      return "\(title) requires verified runtime support for its ordered parts, typed selectors or bounded sample arrangements. Refresh runtime status or select a compatible runtime."
    }
    if let node = graph.nodes.first(where: { RemainingActivationRuntimeCapabilities.operations.contains($0.type) && !remainingActivationRuntimeCapabilities.contains($0.type) }) {
      let title = BundledActivationComponents.entries[node.type]?.definition.title ?? node.type
      return "\(title) requires verified runtime support for its learned slopes, sample direction or seeded training and evaluation behavior. Refresh runtime status or select a compatible runtime."
    }
    if let node = graph.nodes.first(where: { BundledTensorCompositionComponents.entries[$0.type] != nil && !tensorCompositionRuntimeCapabilities.contains($0.type) }) {
      let title = BundledTensorCompositionComponents.entries[node.type]?.definition.title ?? node.type
      return "\(title) requires verified runtime support for its configured directions, source types and separate observations. Refresh runtime status or select a compatible runtime."
    }
    if let node = graph.nodes.first(where: { BundledPoolingComponents.entries[$0.type] != nil && !poolingRuntimeCapabilities.contains($0.type) }) {
      let title = BundledPoolingComponents.entries[node.type]?.definition.title ?? node.type
      return "\(title) requires verified runtime support for pooling windows and typed optional indices. Refresh runtime status or select a compatible runtime."
    }
    if graph.nodes.contains(where: { $0.type == "depthwise_conv2d" }), depthwiseConv2dRuntimeCapability == nil {
      return "DepthwiseConv2d requires verified runtime support for independent filters per input channel and bounded channel multiplication. Refresh runtime status or select a compatible runtime."
    }
    if graph.nodes.contains(where: { $0.type == "conv_transpose3d" }), convTranspose3dRuntimeCapability == nil {
      return "ConvTranspose3d requires verified runtime support for channels-first volume expansion and bounded output-padding triples. Refresh runtime status or select a compatible runtime."
    }
    if graph.nodes.contains(where: { $0.type == "conv_transpose2d" }), convTranspose2dRuntimeCapability == nil {
      return "ConvTranspose2d requires verified runtime support for channels-first spatial expansion and bounded output-padding pairs. Refresh runtime status or select a compatible runtime."
    }
    if graph.nodes.contains(where: { $0.type == "conv_transpose1d" }), convTranspose1dRuntimeCapability == nil {
      return "ConvTranspose1d requires verified runtime support for channels-first transposed convolution, output padding and bounded expansion. Refresh runtime status or select a compatible runtime."
    }
    if graph.nodes.contains(where: { $0.type == "conv3d" }), conv3dRuntimeCapability == nil {
      return "Conv3d requires verified runtime support for channels-first inputs, depth/height/width geometry, grouped kernels and bounded padding. Refresh runtime status or select a compatible runtime."
    }
    if graph.nodes.contains(where: { $0.type == "conv2d" }), conv2dRuntimeCapability == nil {
      return "Conv2d requires verified runtime support for channels-first inputs, paired spatial geometry, grouped kernels and bounded padding. Refresh runtime status or select a compatible runtime."
    }
    if graph.nodes.contains(where: { $0.type == "conv1d" }), conv1dRuntimeCapability == nil {
      return "Conv1d requires verified runtime support for channels-first inputs, grouped kernels and bounded padding. Refresh runtime status or select a compatible runtime."
    }
    if graph.nodes.contains(where: { $0.type == "lazy_linear" }), lazyLinearRuntimeCapability == nil {
      return "LazyLinear requires verified runtime support for inferred widths and initialization before optimizer and state inspection. Refresh runtime status or select a compatible runtime."
    }
    if graph.nodes.contains(where:{$0.type=="mixture_experts"}),!mixtureExpertsRuntimeAvailable{return "Mixture of experts requires the verified routed runtime. Refresh runtime support or select a compatible runtime."}
    if graph.nodes.contains(where: { $0.type == "bilinear" }), bilinearRuntimeCapability == nil {
      return "Bilinear requires verified runtime support for two feature widths, matching sample dimensions, parameter bounds and bias modes. Refresh runtime status or select a compatible runtime."
    }
    if graph.nodes.contains(where: { $0.type == "matmul" }), matrixMultiplyRuntimeCapability == nil {
      return "Matrix multiplication requires verified runtime support for independent batches, sample broadcasting and allocation limits. Refresh runtime status or select a compatible runtime."
    }
    if graph.nodes.contains(where: { $0.type == "softmax" }), softmaxRuntimeCapability == nil {
      return "Softmax requires verified runtime support for explicit sample-axis normalization. Refresh runtime status or select a compatible runtime."
    }
    if graph.nodes.contains(where: { $0.type == "gelu" }), geluRuntimeCapability == nil {
      return "GELU requires a verified runtime capability for exact and tanh approximation modes. Refresh runtime status or select a compatible runtime."
    }
    if let missing = graph.nodes.first(where: { ConfigurableActivationConfiguration.componentIDs.contains($0.type) && !configurableActivationRuntimeCapabilities.contains($0.type) }) {
      let title = registry.component(type: missing.type)?.title ?? missing.type
      return "\(title) requires verified runtime support for its configured float32 activation. Refresh runtime status or select a compatible runtime."
    }
    if graph.nodes.contains(where: { $0.type == "sigmoid" }), sigmoidRuntimeCapability == nil {
      return "Sigmoid requires verified runtime support for a shape-preserving float32 activation with no learned parameters. Refresh runtime status or select a compatible runtime."
    }
    if graph.nodes.contains(where: { TensorLayoutRuntimeCapability.operations.contains($0.type) }), tensorLayoutRuntimeCapability == nil {
      return "Tensor layout components require verified runtime support for reshape, transpose and flatten. Refresh runtime status or select a compatible runtime."
    }
    let errors = validation.issues.filter { $0.severity == .error }
    return errors.isEmpty ? nil : errors.map(\.message).joined(separator: " ")
  }

  /// Accepts a detected exact identity or a bundled manifest pin; never a coverage-catalog version.
  func installRuntimeContract(version: String, expectedVersion: String?) throws {
    runtimeContractVersion = nil
    runtimeExecutionContract = nil
    backendRuntimeReadinessErrors = [:]
    geluRuntimeCapability = nil; sigmoidRuntimeCapability = nil
    layerNormRuntimeCapability = nil; rmsNormRuntimeCapability = nil
    embeddingRuntimeCapability = nil; embeddingBagRuntimeCapabilities = .init()
    matrixMultiplyRuntimeCapability = nil; bilinearRuntimeCapability = nil; self.mixtureExpertsRuntimeAvailable=false; lazyLinearRuntimeCapability = nil; conv1dRuntimeCapability = nil; conv2dRuntimeCapability = nil; conv3dRuntimeCapability = nil; convTranspose1dRuntimeCapability = nil; convTranspose2dRuntimeCapability = nil; convTranspose3dRuntimeCapability = nil; depthwiseConv2dRuntimeCapability = nil; poolingRuntimeCapabilities = .init(); tensorCompositionRuntimeCapabilities = .init(); configurableActivationRuntimeCapabilities = .init(); remainingActivationRuntimeCapabilities = .init(); tensorRoutingRuntimeCapabilities = .init(); spatialPatchRuntimeCapabilities = .init(); statefulNormalizationRuntimeCapabilities = .init(); optimizerRuntimeCapabilities = .init(); schedulerRuntimeCapabilities = .init(); lossRuntimeCapabilities = .init(); recurrentRuntimeCapabilities = .init(); attentionRuntimeCapabilities = .init(); lifecycleRuntimeCapability = .init(); initializationRuntimeCapabilities = .init(); gradientControlRuntimeCapabilities = .init(); dataLoadingRuntimeCapabilities = .init(); programExportRuntimeCapability = .init(); moduleCompositionRuntimeCapability = nil; moduleCompositionDiscoveryEvidence = nil; sklearnRuntimeCapability = nil; tensorflowRuntimeCapability = nil
    softmaxRuntimeCapability = nil
    tensorLayoutRuntimeCapability = nil
    guard
      version.range(of: #"^[0-9]+\.[0-9]+\.[0-9]+[A-Za-z0-9.+_-]*$"#, options: .regularExpression)
        != nil,
      version.trimmingCharacters(in: .whitespacesAndNewlines) == version
    else {
      throw BuiltInCapabilityBridgeError.incompatibleRegistry(
        "The runtime must report an exact PyTorch version, including patch and any build suffix.")
    }
    if let expectedVersion, version != expectedVersion {
      throw BuiltInCapabilityBridgeError.incompatibleRegistry(
        "Loaded PyTorch \(version) differs from the bundled runtime manifest \(expectedVersion).")
    }
    _ = try BuiltInCapabilityBridge.package(registry: registry, runtimeVersion: version)
    runtimeContractVersion = version
  }

  func installDiscoveredRuntimeContract(
    version: String, expectedVersion: String?, devices: [String], descriptor: Data, geluDescriptor: Data? = nil, sigmoidDescriptor: Data? = nil, tensorLayoutDescriptor: Data? = nil, softmaxDescriptor: Data? = nil, layerNormDescriptor: Data? = nil, rmsNormDescriptor: Data? = nil, embeddingDescriptor: Data? = nil, matrixMultiplyDescriptor: Data? = nil, bilinearDescriptor: Data? = nil, mixtureExpertsDescriptor:Data?=nil, lazyLinearDescriptor: Data? = nil, conv1dDescriptor: Data? = nil, conv2dDescriptor: Data? = nil, conv3dDescriptor: Data? = nil, convTranspose1dDescriptor: Data? = nil, convTranspose2dDescriptor: Data? = nil, convTranspose3dDescriptor: Data? = nil, depthwiseConv2dDescriptor: Data? = nil, poolingDescriptors: [String: Data] = [:], tensorCompositionDescriptors: [String: Data] = [:], configurableActivationDescriptors: [String: Data] = [:], remainingActivationDescriptors: [String: Data] = [:], tensorRoutingDescriptors: [String: Data] = [:], spatialPatchDescriptors: [String: Data] = [:], statefulNormalizationDescriptors: [String: Data] = [:], optimizerDescriptors: [String: Data] = [:], schedulerDescriptors: [String: Data] = [:], lossDescriptors: [String: Data] = [:], recurrentDescriptors: [String: Data] = [:], attentionDescriptors: [String: Data] = [:], lifecycleDescriptor: Data? = nil, initializationDescriptor: Data? = nil, gradientPolicyDescriptor: Data? = nil, trainabilityDescriptor: Data? = nil, moduleCompositionDescriptor: Data? = nil, datasetDescriptor: Data? = nil, loaderDescriptor: Data? = nil, programExportDescriptor: Data? = nil, bagDescriptors: [String:Data] = [:]
  ) throws {
    runtimeContractVersion = nil
    runtimeExecutionContract = nil
    backendRuntimeReadinessErrors = [:]
    geluRuntimeCapability = nil; sigmoidRuntimeCapability = nil
    layerNormRuntimeCapability = nil; rmsNormRuntimeCapability = nil
    embeddingRuntimeCapability = nil; embeddingBagRuntimeCapabilities = .init()
    matrixMultiplyRuntimeCapability = nil; bilinearRuntimeCapability = nil; self.mixtureExpertsRuntimeAvailable=false; lazyLinearRuntimeCapability = nil; conv1dRuntimeCapability = nil; conv2dRuntimeCapability = nil; conv3dRuntimeCapability = nil; convTranspose1dRuntimeCapability = nil; convTranspose2dRuntimeCapability = nil; convTranspose3dRuntimeCapability = nil; depthwiseConv2dRuntimeCapability = nil; poolingRuntimeCapabilities = .init(); tensorCompositionRuntimeCapabilities = .init(); configurableActivationRuntimeCapabilities = .init(); remainingActivationRuntimeCapabilities = .init(); tensorRoutingRuntimeCapabilities = .init(); spatialPatchRuntimeCapabilities = .init(); statefulNormalizationRuntimeCapabilities = .init(); optimizerRuntimeCapabilities = .init(); schedulerRuntimeCapabilities = .init(); lossRuntimeCapabilities = .init(); recurrentRuntimeCapabilities = .init(); attentionRuntimeCapabilities = .init(); lifecycleRuntimeCapability = .init(); initializationRuntimeCapabilities = .init(); gradientControlRuntimeCapabilities = .init(); dataLoadingRuntimeCapabilities = .init(); programExportRuntimeCapability = .init(); moduleCompositionRuntimeCapability = nil; moduleCompositionDiscoveryEvidence = nil; sklearnRuntimeCapability = nil; tensorflowRuntimeCapability = nil
    softmaxRuntimeCapability = nil
    tensorLayoutRuntimeCapability = nil
    do {
      guard descriptor.count <= 32_768 else {
        throw BuiltInCapabilityBridgeError.incompatibleRegistry(
          "Worker execution contract exceeds its discovery bound.")
      }
      let typedDevices = devices.compactMap(ContractDevice.init(rawValue:))
      guard typedDevices.count == devices.count else {
        throw BuiltInCapabilityBridgeError.incompatibleRegistry(
          "Worker reported an unsupported execution device.")
      }
      let contract = try JSONDecoder().decode(BuiltInGraphExecutionContract.self, from: descriptor)
      try contract.validate(
        registry: registry, runtimeVersion: version, availableDevices: typedDevices)
      try installRuntimeContract(version: version, expectedVersion: expectedVersion)
      availableDevices = contract.availableDevices.map(\.rawValue)
      runtimeExecutionContract = contract
      embeddingRuntimeCapability = embeddingDescriptor.flatMap { try? EmbeddingRuntimeCapability.decode($0) }
      embeddingBagRuntimeCapabilities = .init(descriptors:bagDescriptors)
      rmsNormRuntimeCapability = rmsNormDescriptor.flatMap {try? RMSNormRuntimeCapability.decode($0)}
      layerNormRuntimeCapability = layerNormDescriptor.flatMap { try? LayerNormRuntimeCapability.decode($0) }
      matrixMultiplyRuntimeCapability = matrixMultiplyDescriptor.flatMap { try? MatrixMultiplyRuntimeCapability.decode($0) }
      conv1dRuntimeCapability = conv1dDescriptor.flatMap { try? Conv1dRuntimeCapability.decode($0) }
      conv2dRuntimeCapability = conv2dDescriptor.flatMap { try? Conv2dRuntimeCapability.decode($0) }
      poolingRuntimeCapabilities = .init(descriptors: poolingDescriptors)
      tensorCompositionRuntimeCapabilities = .init(descriptors: tensorCompositionDescriptors)
      configurableActivationRuntimeCapabilities = .init(descriptors: configurableActivationDescriptors)
      remainingActivationRuntimeCapabilities = .init(descriptors: remainingActivationDescriptors)
      tensorRoutingRuntimeCapabilities = .init(descriptors: tensorRoutingDescriptors)
      spatialPatchRuntimeCapabilities = .init(descriptors: spatialPatchDescriptors)
      statefulNormalizationRuntimeCapabilities = .init(descriptors: statefulNormalizationDescriptors)
      optimizerRuntimeCapabilities = .init(descriptors: optimizerDescriptors)
      schedulerRuntimeCapabilities = .init(descriptors: schedulerDescriptors)
      lossRuntimeCapabilities = .init(descriptors: lossDescriptors)
      recurrentRuntimeCapabilities = .init(descriptors: recurrentDescriptors)
      attentionRuntimeCapabilities = .init(descriptors: attentionDescriptors)
      lifecycleRuntimeCapability = .init(descriptor: lifecycleDescriptor)
      initializationRuntimeCapabilities = .init(descriptor: initializationDescriptor)
      gradientControlRuntimeCapabilities = .init(gradientPolicy: gradientPolicyDescriptor, trainability: trainabilityDescriptor)
      dataLoadingRuntimeCapabilities = .init(dataset:datasetDescriptor,loader:loaderDescriptor)
      programExportRuntimeCapability = .init(descriptor:programExportDescriptor)
      moduleCompositionRuntimeCapability = moduleCompositionDescriptor.flatMap { try? ModuleCompositionRuntimeCapability.decode($0) }
      depthwiseConv2dRuntimeCapability = depthwiseConv2dDescriptor.flatMap { try? DepthwiseConv2dRuntimeCapability.decode($0) }
      convTranspose3dRuntimeCapability = convTranspose3dDescriptor.flatMap { try? ConvTranspose3dRuntimeCapability.decode($0) }
      convTranspose2dRuntimeCapability = convTranspose2dDescriptor.flatMap { try? ConvTranspose2dRuntimeCapability.decode($0) }
      convTranspose1dRuntimeCapability = convTranspose1dDescriptor.flatMap { try? ConvTranspose1dRuntimeCapability.decode($0) }
      conv3dRuntimeCapability = conv3dDescriptor.flatMap { try? Conv3dRuntimeCapability.decode($0) }
      lazyLinearRuntimeCapability = lazyLinearDescriptor.flatMap { try? LazyLinearRuntimeCapability.decode($0) }
      mixtureExpertsRuntimeAvailable=mixtureExpertsDescriptor.map(MixtureExpertsConfiguration.verifyRuntime) ?? false
      bilinearRuntimeCapability = bilinearDescriptor.flatMap { try? BilinearRuntimeCapability.decode($0) }
      softmaxRuntimeCapability = softmaxDescriptor.flatMap { try? SoftmaxRuntimeCapability.decode($0) }
      geluRuntimeCapability = geluDescriptor.flatMap { try? GeluRuntimeCapability.decode($0) }
      sigmoidRuntimeCapability = sigmoidDescriptor.flatMap { try? SigmoidRuntimeCapability.decode($0) }
      tensorLayoutRuntimeCapability = tensorLayoutDescriptor.flatMap { try? TensorLayoutRuntimeCapability.decode($0) }
    } catch {
      runtimeContractVersion = nil
      runtimeExecutionContract = nil
      backendRuntimeReadinessErrors = [:]
      geluRuntimeCapability = nil; sigmoidRuntimeCapability = nil
      layerNormRuntimeCapability = nil; rmsNormRuntimeCapability = nil
      embeddingRuntimeCapability = nil; embeddingBagRuntimeCapabilities = .init()
      matrixMultiplyRuntimeCapability = nil; bilinearRuntimeCapability = nil; self.mixtureExpertsRuntimeAvailable=false; lazyLinearRuntimeCapability = nil; conv1dRuntimeCapability = nil; conv2dRuntimeCapability = nil; conv3dRuntimeCapability = nil; convTranspose1dRuntimeCapability = nil; convTranspose2dRuntimeCapability = nil; convTranspose3dRuntimeCapability = nil; depthwiseConv2dRuntimeCapability = nil; poolingRuntimeCapabilities = .init(); tensorCompositionRuntimeCapabilities = .init(); configurableActivationRuntimeCapabilities = .init(); remainingActivationRuntimeCapabilities = .init(); tensorRoutingRuntimeCapabilities = .init(); spatialPatchRuntimeCapabilities = .init(); statefulNormalizationRuntimeCapabilities = .init(); optimizerRuntimeCapabilities = .init(); schedulerRuntimeCapabilities = .init(); lossRuntimeCapabilities = .init(); recurrentRuntimeCapabilities = .init(); attentionRuntimeCapabilities = .init(); lifecycleRuntimeCapability = .init(); initializationRuntimeCapabilities = .init(); gradientControlRuntimeCapabilities = .init(); dataLoadingRuntimeCapabilities = .init(); programExportRuntimeCapability = .init(); moduleCompositionRuntimeCapability = nil; moduleCompositionDiscoveryEvidence = nil; sklearnRuntimeCapability = nil; tensorflowRuntimeCapability = nil
      softmaxRuntimeCapability = nil
      tensorLayoutRuntimeCapability = nil
      throw error
    }
  }

  /// Readiness belongs to this project window, not another window's Settings.
  func estimatorRuntimeReadinessMessage(backend: String) -> String? {
    if capabilityOperation != nil || pendingCapabilityOperation != nil {
      return "Checking runtime support for \(backend)… Wait for this check to finish, then run again."
    }
    if let failure = backendRuntimeReadinessErrors[backend] { return failure }
    if (backend == "sklearn" && sklearnRuntimeCapability != nil) ||
        (backend == "tensorflow" && tensorflowRuntimeCapability != nil) { return nil }
    return "This runtime does not provide verified \(backend) support. Refresh runtime status or select a complete application build."
  }

  func refreshCapabilities(completion: ((Bool) -> Void)? = nil) {
    let token = UUID()
    capabilityProbeToken = token
    runtimeContractVersion = nil; runtimeExecutionContract = nil
    backendRuntimeReadinessErrors = [:]
    geluRuntimeCapability = nil; sigmoidRuntimeCapability = nil; layerNormRuntimeCapability = nil; rmsNormRuntimeCapability = nil; embeddingRuntimeCapability = nil; embeddingBagRuntimeCapabilities = .init()
    matrixMultiplyRuntimeCapability = nil; bilinearRuntimeCapability = nil; self.mixtureExpertsRuntimeAvailable=false; lazyLinearRuntimeCapability = nil; conv1dRuntimeCapability = nil; conv2dRuntimeCapability = nil; conv3dRuntimeCapability = nil; convTranspose1dRuntimeCapability = nil; convTranspose2dRuntimeCapability = nil; convTranspose3dRuntimeCapability = nil; depthwiseConv2dRuntimeCapability = nil; poolingRuntimeCapabilities = .init(); tensorCompositionRuntimeCapabilities = .init(); configurableActivationRuntimeCapabilities = .init(); remainingActivationRuntimeCapabilities = .init(); tensorRoutingRuntimeCapabilities = .init(); spatialPatchRuntimeCapabilities = .init(); statefulNormalizationRuntimeCapabilities = .init(); optimizerRuntimeCapabilities = .init(); schedulerRuntimeCapabilities = .init(); lossRuntimeCapabilities = .init(); recurrentRuntimeCapabilities = .init(); attentionRuntimeCapabilities = .init(); lifecycleRuntimeCapability = .init(); initializationRuntimeCapabilities = .init(); gradientControlRuntimeCapabilities = .init(); dataLoadingRuntimeCapabilities = .init(); programExportRuntimeCapability = .init(); moduleCompositionRuntimeCapability = nil; moduleCompositionDiscoveryEvidence = nil; sklearnRuntimeCapability = nil; tensorflowRuntimeCapability = nil; softmaxRuntimeCapability = nil; tensorLayoutRuntimeCapability = nil
    gradientCapture.installCapability(nil)
    invalidateDraftCode()
    let previousPending = pendingCapabilityOperation
    pendingCapabilityOperation = nil
    guard let runtime else {
      capabilityStatus = runtimeError ?? "Runtime unavailable."
      cancelActiveCapabilityProbe()
      previousPending?.complete(false); completion?(false)
      return
    }
    do {
      let policy = try executionPolicy(for: nil)
      let window = try ExecutionBudgetWindow(policy: policy.budget, startedAtUptime: ProcessInfo.processInfo.systemUptime)
      let operation = CapabilityDiscoveryOperation(token: token, runtime: runtime, window: window,
          storageLimitBytes: policy.storageLimitBytes, runtimeEvidence: try? WorkflowRuntimeEvidence.capture(runtime), completion: completion)
      pendingCapabilityOperation = operation
      capabilityStatus = "Checking exact runtime identity…"
      cancelActiveCapabilityProbe()
      previousPending?.complete(false)
      launchPendingCapabilityProbe()
    } catch {
      cancelActiveCapabilityProbe(); previousPending?.complete(false)
      capabilityStatus = error.localizedDescription; completion?(false)
    }
  }
  private func cancelActiveCapabilityProbe() {
    guard let operation = capabilityOperation, !operation.cancelled else { return }
    operation.cancelled = true; operation.launchTask?.cancel(); operation.worker?.cancel()
  }
  private func launchPendingCapabilityProbe() {
    guard capabilityOperation == nil, let operation = pendingCapabilityOperation else { return }
    pendingCapabilityOperation = nil; capabilityOperation = operation
    let line: (String) -> Void = { [self] line in
      guard capabilityOperation === operation, !operation.finished, !operation.cancelled,
            operation.token == capabilityProbeToken, operation.failure == nil else { return }
      guard let data = line.data(using: .utf8), data.count <= 1_000_000,
            let value = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
            value["event"] as? String == "capabilities", operation.result == nil else {
        operation.failure = "Runtime detection returned an invalid or duplicate capability event."; return
      }
      operation.result = value
    }
    let exit: (Int32, String) -> Void = { [self] code, stderr in finishCapabilityProbe(operation, code: code, stderr: stderr) }
    func install(_ handle: WorkflowWorkerHandle) {
      guard !operation.finished else { return }
      operation.worker = handle
      if operation.cancelled { handle.cancel() }
    }
    if let capabilityLaunch {
      do {
        _ = try operation.window.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
        install(try capabilityLaunch(workspaceDirectory, capabilityDiscoveryScope, operation.runtime, operation.window, line, exit))
      } catch { finishCapabilityProbe(operation, code: 1, stderr: error.localizedDescription) }
    } else {
      operation.launchTask = Task { [self] in
        defer { operation.launchTask = nil }
        do {
          try Task.checkCancellation()
          let handle = try await WorkflowToolExecution.launchAsync(workspace: workspaceDirectory, accountingRoot: applicationDirectory, projectID: capabilityDiscoveryScope,
            runtime: operation.runtime, window: operation.window, toolMode: .runtimeCapabilities, inputs: [:],
            maximumSeconds: Double(operation.window.policy.maximumRunSeconds), storageLimitBytes: operation.storageLimitBytes,
            onLine: line, onExit: exit)
          install(handle)
        } catch { finishCapabilityProbe(operation, code: 1, stderr: error.localizedDescription) }
      }
    }
  }
  private func finishCapabilityProbe(_ operation: CapabilityDiscoveryOperation, code: Int32, stderr: String) {
    guard !operation.finished else { return }
    operation.finished = true; operation.worker = nil
    if capabilityOperation === operation { capabilityOperation = nil }
    defer { launchPendingCapabilityProbe() }
    guard operation.token == capabilityProbeToken, !operation.cancelled else { operation.complete(false); return }
    let runtime = operation.runtime
    do {
      _ = try operation.window.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
      guard code == 0, operation.failure == nil, let event = operation.result, let version = event["torchVersion"] as? String,
        let devices = event["devices"] as? [String], !devices.isEmpty,
        devices.allSatisfy({ ["cpu", "mps"].contains($0) }),
        let defaultDevice = event["defaultDevice"] as? String, devices.contains(defaultDevice)
      else {
        throw BuiltInCapabilityBridgeError.incompatibleRegistry(
          operation.failure ?? (stderr.isEmpty
            ? "Runtime detection did not finish with a valid version and device contract."
            : "Runtime detection failed: \(stderr.prefix(200))"))
      }
      guard let descriptor = event["executionContract"] as? [String: Any] else {
        throw BuiltInCapabilityBridgeError.incompatibleRegistry(
          "Worker did not report its execution contract. Rebuild or select a compatible local runtime."
        )
      }
      try self.installDiscoveredRuntimeContract(
        version: version, expectedVersion: runtime.packageVersions["torch"], devices: devices,
        descriptor: JSONSerialization.data(withJSONObject: descriptor),
        geluDescriptor: ((event["componentCapabilities"] as? [String: Any])?["gelu"]).flatMap { try? JSONSerialization.data(withJSONObject: $0) },
        sigmoidDescriptor: ((event["componentCapabilities"] as? [String: Any])?["sigmoid"]).flatMap { try? JSONSerialization.data(withJSONObject: $0) },
        tensorLayoutDescriptor: ((event["componentCapabilities"] as? [String: Any])?["tensorLayout"]).flatMap { try? JSONSerialization.data(withJSONObject: $0) },
        softmaxDescriptor: ((event["componentCapabilities"] as? [String: Any])?["softmax"]).flatMap { try? JSONSerialization.data(withJSONObject: $0) },
        layerNormDescriptor: ((event["componentCapabilities"] as? [String: Any])?["layerNorm"]).flatMap { try? JSONSerialization.data(withJSONObject: $0) },
        rmsNormDescriptor: ((event["componentCapabilities"] as? [String: Any])?["rms_norm"]).flatMap {try? JSONSerialization.data(withJSONObject:$0)},
        embeddingDescriptor: ((event["componentCapabilities"] as? [String: Any])?["embedding"]).flatMap { try? JSONSerialization.data(withJSONObject: $0) },
        matrixMultiplyDescriptor: ((event["componentCapabilities"] as? [String: Any])?["matrixMultiply"]).flatMap { try? JSONSerialization.data(withJSONObject: $0) },
        bilinearDescriptor: ((event["componentCapabilities"] as? [String: Any])?["bilinear"]).flatMap { try? JSONSerialization.data(withJSONObject: $0) },
        mixtureExpertsDescriptor: ((event["componentCapabilities"] as? [String:Any])?["mixture_experts"]).flatMap{try? JSONSerialization.data(withJSONObject:$0)},
        lazyLinearDescriptor: ((event["componentCapabilities"] as? [String: Any])?["lazyLinear"]).flatMap { try? JSONSerialization.data(withJSONObject: $0) },
        conv1dDescriptor: ((event["componentCapabilities"] as? [String: Any])?["conv1d"]).flatMap { try? JSONSerialization.data(withJSONObject: $0) },
        conv2dDescriptor: ((event["componentCapabilities"] as? [String: Any])?["conv2d"]).flatMap { try? JSONSerialization.data(withJSONObject: $0) },
        conv3dDescriptor: ((event["componentCapabilities"] as? [String: Any])?["conv3d"]).flatMap { try? JSONSerialization.data(withJSONObject: $0) },
        convTranspose1dDescriptor: ((event["componentCapabilities"] as? [String: Any])?["conv_transpose1d"]).flatMap { try? JSONSerialization.data(withJSONObject: $0) },
        convTranspose2dDescriptor: ((event["componentCapabilities"] as? [String: Any])?["conv_transpose2d"]).flatMap { try? JSONSerialization.data(withJSONObject: $0) },
        convTranspose3dDescriptor: ((event["componentCapabilities"] as? [String: Any])?["conv_transpose3d"]).flatMap { try? JSONSerialization.data(withJSONObject: $0) },
        depthwiseConv2dDescriptor: ((event["componentCapabilities"] as? [String: Any])?["depthwise_conv2d"]).flatMap { try? JSONSerialization.data(withJSONObject: $0) },
        poolingDescriptors: (event["componentCapabilities"] as? [String: Any] ?? [:]).filter { BundledPoolingComponents.entries[$0.key] != nil }.compactMapValues { try? JSONSerialization.data(withJSONObject: $0) },
        tensorCompositionDescriptors: (event["componentCapabilities"] as? [String: Any] ?? [:]).filter { BundledTensorCompositionComponents.entries[$0.key] != nil }.compactMapValues { try? JSONSerialization.data(withJSONObject: $0) },
        configurableActivationDescriptors: (event["componentCapabilities"] as? [String: Any] ?? [:]).filter { ConfigurableActivationConfiguration.componentIDs.contains($0.key) }.compactMapValues { try? JSONSerialization.data(withJSONObject: $0) },
        remainingActivationDescriptors: (event["componentCapabilities"] as? [String: Any] ?? [:]).filter { RemainingActivationRuntimeCapabilities.operations.contains($0.key) }.compactMapValues { try? JSONSerialization.data(withJSONObject: $0) },
        tensorRoutingDescriptors: (event["componentCapabilities"] as? [String: Any] ?? [:]).filter { TensorRoutingRuntimeCapabilities.operations.contains($0.key) }.compactMapValues { try? JSONSerialization.data(withJSONObject: $0) },
        spatialPatchDescriptors: (event["componentCapabilities"] as? [String: Any] ?? [:]).filter { SpatialPatchRuntimeCapabilities.operations.contains($0.key) }.compactMapValues { try? JSONSerialization.data(withJSONObject: $0) },
        statefulNormalizationDescriptors: (event["componentCapabilities"] as? [String: Any] ?? [:]).filter { StatefulNormalizationRuntimeCapabilities.operations.contains($0.key) }.compactMapValues { try? JSONSerialization.data(withJSONObject: $0) },
        optimizerDescriptors: (event["componentCapabilities"] as? [String: Any] ?? [:]).filter { OptimizerConfiguration.componentIDs.contains($0.key) }.compactMapValues { try? JSONSerialization.data(withJSONObject: $0) },
        schedulerDescriptors: (event["componentCapabilities"] as? [String: Any] ?? [:]).filter { SchedulerConfiguration.componentIDs.contains($0.key) }.compactMapValues { try? JSONSerialization.data(withJSONObject: $0) },
        lossDescriptors: (event["componentCapabilities"] as? [String: Any] ?? [:]).filter { LossConfiguration.componentIDs.contains($0.key) }.compactMapValues { try? JSONSerialization.data(withJSONObject: $0) },
        recurrentDescriptors: (event["componentCapabilities"] as? [String: Any] ?? [:]).filter { RecurrentConfiguration.componentIDs.contains($0.key) }.compactMapValues { try? JSONSerialization.data(withJSONObject: $0) },
        attentionDescriptors: (event["componentCapabilities"] as? [String: Any] ?? [:]).filter { AttentionConfiguration.componentIDs.contains($0.key) }.compactMapValues { try? JSONSerialization.data(withJSONObject: $0) },
        lifecycleDescriptor: ((event["lifecycleCapabilities"] as? [String: Any])?["checkpoint_lifecycle"]).flatMap { try? JSONSerialization.data(withJSONObject: $0) },
        initializationDescriptor: ((event["initializationCapabilities"] as? [String: Any])?["initialization"]).flatMap { try? JSONSerialization.data(withJSONObject: $0) },
        gradientPolicyDescriptor: ((event["gradientPolicyCapabilities"] as? [String: Any])?["gradient_policy"]).flatMap { try? JSONSerialization.data(withJSONObject: $0) },
        trainabilityDescriptor: ((event["trainabilityCapabilities"] as? [String: Any])?["trainability"]).flatMap { try? JSONSerialization.data(withJSONObject: $0) },
        moduleCompositionDescriptor: ((event["compositionCapabilities"] as? [String: Any])?["moduleComposition"]).flatMap { try? JSONSerialization.data(withJSONObject: $0) },
        datasetDescriptor: ((event["dataLoadingCapabilities"] as? [String:Any])?["dataset_source"]).flatMap {try? JSONSerialization.data(withJSONObject:$0)},
        loaderDescriptor: ((event["dataLoadingCapabilities"] as? [String:Any])?["data_loader"]).flatMap {try? JSONSerialization.data(withJSONObject:$0)},
        programExportDescriptor: ((event["programExportCapabilities"] as? [String:Any])?["program_export"]).flatMap {try? JSONSerialization.data(withJSONObject:$0)},
        bagDescriptors: (event["componentCapabilities"] as? [String:Any] ?? [:]).filter {EmbeddingBagRuntimeCapabilities.operations.contains($0.key)}.compactMapValues {try? JSONSerialization.data(withJSONObject:$0)})
      if let capability = self.moduleCompositionRuntimeCapability {
        guard let before = operation.runtimeEvidence,
              self.runtime == runtime,
              try WorkflowRuntimeEvidence.capture(runtime) == before else {
          throw GraphConfigurationError("The runtime changed while composition support was being checked. Refresh the runtime.")
        }
        self.moduleCompositionDiscoveryEvidence = ModuleCompositionDiscoveryEvidence(
          capability: capability, runtime: runtime, runtimeEvidence: before)
      }
      let optional = event["optionalBackends"] as? [String: Any] ?? [:]
      let optionalErrors = event["optionalBackendErrors"] as? [String: Any] ?? [:]
      self.sklearnRuntimeCapability = optional["sklearn"].flatMap { try? JSONSerialization.data(withJSONObject: $0) }.flatMap { try? SklearnRuntimeCapabilities.decode($0) }
      self.tensorflowRuntimeCapability = optional["tensorflow"].flatMap { try? JSONSerialization.data(withJSONObject: $0) }.flatMap { try? TensorFlowRuntimeCapabilities.decode($0) }
      for backend in ["sklearn", "tensorflow"] {
        let verified = backend == "sklearn" ? self.sklearnRuntimeCapability != nil : self.tensorflowRuntimeCapability != nil
        if optionalErrors[backend] != nil {
          // Worker exception text may contain private paths. Retain only fixed,
          // actionable copy, and refuse contradictory success/error descriptors.
          self.backendRuntimeReadinessErrors[backend] = "The runtime could not load \(backend). Refresh runtime status or reinstall the application."
          if backend == "sklearn" { self.sklearnRuntimeCapability = nil }
          else { self.tensorflowRuntimeCapability = nil }
        } else if optional[backend] != nil && !verified {
          self.backendRuntimeReadinessErrors[backend] = "The runtime reported incompatible \(backend) support. Refresh runtime status or select a compatible application build."
        }
      }
      self.gradientCapture.installCapability(event["gradientCapture"].flatMap { try? JSONSerialization.data(withJSONObject: $0) })
      self.capabilityStatus =
        "PyTorch \(version) · "
        + devices.map { $0 == "mps" ? "Apple GPU (MPS)" : "CPU" }.joined(separator: ", ")
        + " · \(self.verifiedRuntimeComponentCount) verified components"
      operation.complete(true)
      if self.activeSection == "Code" { self.refreshDraftCode() }
    } catch {
      self.runtimeContractVersion = nil
      self.runtimeExecutionContract = nil
      self.geluRuntimeCapability = nil; self.sigmoidRuntimeCapability = nil
      self.layerNormRuntimeCapability = nil; self.rmsNormRuntimeCapability = nil
      self.embeddingRuntimeCapability = nil; self.embeddingBagRuntimeCapabilities = .init()
      self.matrixMultiplyRuntimeCapability = nil; self.bilinearRuntimeCapability = nil; self.mixtureExpertsRuntimeAvailable=false; self.mixtureExpertsRuntimeAvailable=false; self.lazyLinearRuntimeCapability = nil; self.conv1dRuntimeCapability = nil; self.conv2dRuntimeCapability = nil; self.conv3dRuntimeCapability = nil; self.convTranspose1dRuntimeCapability = nil; self.convTranspose2dRuntimeCapability = nil; self.convTranspose3dRuntimeCapability = nil; self.depthwiseConv2dRuntimeCapability = nil; self.poolingRuntimeCapabilities = .init(); self.tensorCompositionRuntimeCapabilities = .init(); self.configurableActivationRuntimeCapabilities = .init(); self.remainingActivationRuntimeCapabilities = .init(); self.tensorRoutingRuntimeCapabilities = .init(); self.spatialPatchRuntimeCapabilities = .init(); self.statefulNormalizationRuntimeCapabilities = .init(); self.optimizerRuntimeCapabilities = .init(); self.schedulerRuntimeCapabilities = .init(); self.lossRuntimeCapabilities = .init(); self.recurrentRuntimeCapabilities = .init(); self.attentionRuntimeCapabilities = .init(); self.lifecycleRuntimeCapability = .init(); self.initializationRuntimeCapabilities = .init(); self.gradientControlRuntimeCapabilities = .init(); self.dataLoadingRuntimeCapabilities = .init(); self.programExportRuntimeCapability = .init(); self.moduleCompositionRuntimeCapability = nil; self.moduleCompositionDiscoveryEvidence = nil; self.sklearnRuntimeCapability = nil; self.tensorflowRuntimeCapability = nil
      self.softmaxRuntimeCapability = nil
      self.tensorLayoutRuntimeCapability = nil
      self.capabilityStatus = error.localizedDescription
      operation.complete(false)
    }
  }
}

// Each queued request captures its own budget. Only one discovery worker may import the runtime at once.
typealias CapabilityDiscoveryLaunch = @MainActor (URL, String, RuntimeLocation, ExecutionBudgetWindow,
    @escaping (String) -> Void, @escaping (Int32, String) -> Void) throws -> WorkflowWorkerHandle

@MainActor final class CapabilityDiscoveryOperation {
    let token: UUID
    let runtime: RuntimeLocation
    let window: ExecutionBudgetWindow
    let storageLimitBytes: Int64?
    let runtimeEvidence: WorkflowRuntimeEvidence?
    var launchTask: Task<Void, Never>?
    var worker: WorkflowWorkerHandle?
    var result: [String: Any]?
    var failure: String?
    var cancelled = false
    var finished = false
    private var completion: ((Bool) -> Void)?
    init(token: UUID, runtime: RuntimeLocation, window: ExecutionBudgetWindow, storageLimitBytes: Int64? = nil, runtimeEvidence: WorkflowRuntimeEvidence? = nil, completion: ((Bool) -> Void)?) {
      self.token = token; self.runtime = runtime; self.window = window; self.storageLimitBytes = storageLimitBytes; self.runtimeEvidence = runtimeEvidence; self.completion = completion
    }
    func complete(_ success: Bool) {
      let callback = completion; completion = nil; callback?(success)
    }
}

/// Nonpersisted proof from an owned discovery worker, qualified by its complete runtime inventory.
/// Decoding a descriptor or installing one manually cannot establish this proof.
struct ModuleCompositionDiscoveryEvidence {
    let capability: ModuleCompositionRuntimeCapability
    let runtime: RuntimeLocation
    let runtimeEvidence: WorkflowRuntimeEvidence
    fileprivate init(capability: ModuleCompositionRuntimeCapability, runtime: RuntimeLocation,
                     runtimeEvidence: WorkflowRuntimeEvidence) {
        self.capability = capability; self.runtime = runtime; self.runtimeEvidence = runtimeEvidence
    }
    func verifiedDescriptor(for runtime: RuntimeLocation, evidence: WorkflowRuntimeEvidence) throws -> JSONValue {
        guard self.runtime == runtime, runtimeEvidence == evidence else {
            throw GraphConfigurationError("Composition support was verified for a different runtime. Refresh the runtime.")
        }
        return capability.descriptor
    }
}
