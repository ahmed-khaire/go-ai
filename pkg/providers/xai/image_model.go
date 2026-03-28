package xai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"

	"github.com/digitallysavvy/go-ai/pkg/internal/fileutil"
	"github.com/digitallysavvy/go-ai/pkg/provider"
	providererrors "github.com/digitallysavvy/go-ai/pkg/provider/errors"
	"github.com/digitallysavvy/go-ai/pkg/provider/types"
)

// ImageModel implements the provider.ImageModel interface for XAI
type ImageModel struct {
	provider *Provider
	modelID  string
}

// NewImageModel creates a new XAI image generation model
func NewImageModel(prov *Provider, modelID string) *ImageModel {
	return &ImageModel{
		provider: prov,
		modelID:  modelID,
	}
}

// SpecificationVersion returns the specification version
func (m *ImageModel) SpecificationVersion() string {
	return "v3"
}

// Provider returns the provider name
func (m *ImageModel) Provider() string {
	return "xai"
}

// ModelID returns the model ID
func (m *ImageModel) ModelID() string {
	return m.modelID
}

// XAIImageProviderOptions contains provider-specific options for XAI image generation
type XAIImageProviderOptions struct {
	// AspectRatio for image generation (e.g., "16:9", "1:1", "9:16")
	AspectRatio *string `json:"aspect_ratio,omitempty"`

	// OutputFormat specifies the image format.
	// Valid values: "png", "jpeg", "b64_json".
	// Use "b64_json" to receive base64-encoded image data instead of a URL.
	OutputFormat *string `json:"output_format,omitempty"`

	// SyncMode controls synchronous vs asynchronous generation
	SyncMode *bool `json:"sync_mode,omitempty"`

	// Resolution controls the output resolution tier of the generated image.
	// Accepted values: "1k" (1024px), "2k" (2048px).
	// Only supported by models that accept this option (e.g., grok-imagine-image-pro).
	Resolution *string `json:"resolution,omitempty"`

	// Quality controls the output quality.
	// Valid values: "low", "medium", "high".
	Quality *string `json:"quality,omitempty"`

	// User is a unique identifier for the end user, used for abuse detection.
	User *string `json:"user,omitempty"`
}

// XAIImageMetadata holds provider-specific metadata returned by XAI image models.
type XAIImageMetadata struct {
	// Images contains per-image metadata (e.g., revised prompts).
	Images []XAIImageItemMetadata `json:"images,omitempty"`

	// CostInUsdTicks is the cost of the image generation in USD ticks
	// (1 tick = 0.000001 USD).
	CostInUsdTicks *int64 `json:"costInUsdTicks,omitempty"`
}

// XAIImageItemMetadata holds per-image metadata from the XAI image API.
type XAIImageItemMetadata struct {
	// RevisedPrompt is the prompt that was actually used to generate the image,
	// after any safety or quality revisions applied by the model.
	RevisedPrompt *string `json:"revisedPrompt,omitempty"`
}

// DoGenerate performs image generation or editing
func (m *ImageModel) DoGenerate(ctx context.Context, opts *provider.ImageGenerateOptions) (*types.ImageResult, error) {
	warnings := []types.Warning{}

	// Extract provider options
	provOpts, err := extractImageProviderOptions(opts.ProviderOptions)
	if err != nil {
		return nil, err
	}

	// Check for unsupported options
	warnings = append(warnings, m.checkUnsupportedOptions(opts)...)

	// Determine if this is editing or generation
	hasFiles := len(opts.Files) > 0
	endpoint := "/v1/images/generations"
	if hasFiles {
		endpoint = "/v1/images/edits"
	}

	// Build request body
	body := m.buildRequestBody(opts, provOpts, hasFiles)

	// Make API request
	var resp xaiImageResponse
	if err := m.provider.client.PostJSON(ctx, endpoint, body, &resp); err != nil {
		return nil, m.handleError(err)
	}

	// Check if we have at least one image
	if len(resp.Data) == 0 {
		return nil, providererrors.NewProviderError("xai", 0, "",
			"no images in response", nil)
	}

	imageData := resp.Data[0]

	// Handle b64_json or URL response format.
	var imageBytes []byte
	var mimeType string
	if imageData.B64JSON != "" {
		decoded, decErr := base64.StdEncoding.DecodeString(imageData.B64JSON)
		if decErr != nil {
			return nil, providererrors.NewProviderError("xai", 0, "",
				fmt.Sprintf("failed to decode b64_json image: %v", decErr), decErr)
		}
		imageBytes = decoded
		mimeType = "image/png"
	} else if imageData.URL != "" {
		imageBytes, err = m.downloadImage(ctx, imageData.URL)
		if err != nil {
			return nil, providererrors.NewProviderError("xai", 0, "",
				fmt.Sprintf("failed to download image: %v", err), err)
		}
		mimeType = "image/png"
	} else {
		return nil, providererrors.NewProviderError("xai", 0, "",
			"no image data (neither url nor b64_json) in response", nil)
	}

	// Build result with single image
	result := &types.ImageResult{
		Image:    imageBytes,
		MimeType: mimeType,
		URL:      imageData.URL,
		Usage: types.ImageUsage{
			ImageCount: len(resp.Data),
		},
		Warnings: warnings,
	}

	// Always build providerMetadata with per-image array (exposes revisedPrompt).
	imagesMeta := make([]XAIImageItemMetadata, len(resp.Data))
	for i, d := range resp.Data {
		imagesMeta[i] = XAIImageItemMetadata{RevisedPrompt: d.RevisedPrompt}
	}
	meta := XAIImageMetadata{Images: imagesMeta}
	if resp.Usage != nil && resp.Usage.CostInUsdTicks != nil {
		meta.CostInUsdTicks = resp.Usage.CostInUsdTicks
	}
	result.ProviderMetadata = map[string]interface{}{"xai": meta}

	return result, nil
}

// buildRequestBody constructs the API request body
func (m *ImageModel) buildRequestBody(opts *provider.ImageGenerateOptions, provOpts *XAIImageProviderOptions, hasFiles bool) map[string]interface{} {
	body := map[string]interface{}{
		"model":           m.modelID,
		"prompt":          opts.Prompt,
		"response_format": "b64_json", // Always request base64 data directly from the API.
	}

	// Add N (number of images)
	n := 1
	if opts.N != nil && *opts.N > 0 {
		n = *opts.N
	}
	body["n"] = n

	// Add aspect ratio (prefer standard option over provider option)
	if opts.AspectRatio != "" {
		body["aspect_ratio"] = opts.AspectRatio
	} else if provOpts.AspectRatio != nil {
		body["aspect_ratio"] = *provOpts.AspectRatio
	}

	// Add provider-specific options
	if provOpts.OutputFormat != nil {
		body["output_format"] = *provOpts.OutputFormat
	}

	if provOpts.SyncMode != nil {
		body["sync_mode"] = *provOpts.SyncMode
	}

	if provOpts.Resolution != nil {
		body["resolution"] = *provOpts.Resolution
	}

	if provOpts.Quality != nil {
		body["quality"] = *provOpts.Quality
	}

	if provOpts.User != nil {
		body["user"] = *provOpts.User
	}

	// Add source images for editing — accepts multiple reference images as an array.
	if hasFiles {
		images := make([]map[string]interface{}, 0, len(opts.Files))
		for _, f := range opts.Files {
			images = append(images, map[string]interface{}{
				"url":  m.convertImageFileToDataURI(f),
				"type": "image_url",
			})
		}
		body["images"] = images
	}

	// mask is not supported by the xAI image API — omitted intentionally.

	return body
}

// convertImageFileToDataURI converts an ImageFile to a data URI
func (m *ImageModel) convertImageFileToDataURI(file provider.ImageFile) string {
	if file.Type == "url" {
		return file.URL
	}

	// Convert binary data to base64 data URL
	base64Data := base64.StdEncoding.EncodeToString(file.Data)
	mediaType := file.MediaType
	if mediaType == "" {
		mediaType = "image/png"
	}

	return fmt.Sprintf("data:%s;base64,%s", mediaType, base64Data)
}

// checkUnsupportedOptions checks for unsupported options and generates warnings
func (m *ImageModel) checkUnsupportedOptions(opts *provider.ImageGenerateOptions) []types.Warning {
	warnings := []types.Warning{}

	if opts.Size != "" {
		warnings = append(warnings, types.Warning{
			Type:    "unsupported-option",
			Message: "XAI image model does not support the 'size' option. Use 'aspectRatio' instead.",
		})
	}

	if opts.Seed != nil {
		warnings = append(warnings, types.Warning{
			Type:    "unsupported-option",
			Message: "XAI image model does not support seed",
		})
	}

	if opts.Mask != nil {
		warnings = append(warnings, types.Warning{
			Type:    "unsupported-option",
			Message: "XAI image model does not support mask/inpainting.",
		})
	}

	return warnings
}

// downloadImage downloads an image from a URL with size limits to prevent DoS
func (m *ImageModel) downloadImage(ctx context.Context, url string) ([]byte, error) {
	return fileutil.Download(ctx, url, fileutil.DefaultDownloadOptions())
}

// extractImageProviderOptions extracts XAI-specific provider options
func extractImageProviderOptions(opts map[string]interface{}) (*XAIImageProviderOptions, error) {
	if opts == nil {
		return &XAIImageProviderOptions{}, nil
	}

	xaiOpts, ok := opts["xai"]
	if !ok {
		return &XAIImageProviderOptions{}, nil
	}

	// Convert to JSON and back to struct
	jsonData, err := json.Marshal(xaiOpts)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal provider options: %w", err)
	}

	var provOpts XAIImageProviderOptions
	if err := json.Unmarshal(jsonData, &provOpts); err != nil {
		return nil, fmt.Errorf("failed to unmarshal provider options: %w", err)
	}

	return &provOpts, nil
}

// handleError converts provider errors
func (m *ImageModel) handleError(err error) error {
	if provErr, ok := err.(*providererrors.ProviderError); ok {
		return provErr
	}
	return providererrors.NewProviderError("xai", 0, "", err.Error(), err)
}

// xaiImageResponse represents the image generation API response
type xaiImageResponse struct {
	Data  []xaiImageData  `json:"data"`
	Usage *xaiImageUsage  `json:"usage,omitempty"`
}

// xaiImageUsage holds top-level usage data from the image generation response.
type xaiImageUsage struct {
	CostInUsdTicks *int64 `json:"cost_in_usd_ticks,omitempty"`
}

// xaiImageData represents image data in the response
type xaiImageData struct {
	URL           string  `json:"url,omitempty"`
	B64JSON       string  `json:"b64_json,omitempty"`
	RevisedPrompt *string `json:"revised_prompt,omitempty"`
}
