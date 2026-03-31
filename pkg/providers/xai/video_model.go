package xai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"time"

	"github.com/digitallysavvy/go-ai/pkg/internal/polling"
	"github.com/digitallysavvy/go-ai/pkg/provider"
	providererrors "github.com/digitallysavvy/go-ai/pkg/provider/errors"
	"github.com/digitallysavvy/go-ai/pkg/provider/types"
)

// VideoModel implements the provider.VideoModelV3 interface for XAI
type VideoModel struct {
	provider *Provider
	modelID  string
}

// NewVideoModel creates a new XAI video generation model
func NewVideoModel(prov *Provider, modelID string) *VideoModel {
	return &VideoModel{
		provider: prov,
		modelID:  modelID,
	}
}

// SpecificationVersion returns the specification version
func (m *VideoModel) SpecificationVersion() string {
	return "v3"
}

// Provider returns the provider name
func (m *VideoModel) Provider() string {
	return "xai"
}

// ModelID returns the model ID
func (m *VideoModel) ModelID() string {
	return m.modelID
}

// MaxVideosPerCall returns nil (XAI generates one video per call)
func (m *VideoModel) MaxVideosPerCall() *int {
	maxVideos := 1
	return &maxVideos
}

// XAIVideoProviderOptions contains provider-specific options for XAI video generation
type XAIVideoProviderOptions struct {
	// PollIntervalMs is the interval between status checks in milliseconds (default: 5000)
	PollIntervalMs *int `json:"pollIntervalMs,omitempty"`

	// PollTimeoutMs is the maximum time to wait for video generation in milliseconds (default: 600000)
	PollTimeoutMs *int `json:"pollTimeoutMs,omitempty"`

	// Resolution is the output resolution: "480p" or "720p"
	Resolution *string `json:"resolution,omitempty"`

	// VideoURL is the source video URL for video editing
	VideoURL *string `json:"videoUrl,omitempty"`
}

// DoGenerate performs video generation with polling
func (m *VideoModel) DoGenerate(ctx context.Context, opts *provider.VideoModelV3CallOptions) (*provider.VideoModelV3Response, error) {
	warnings := []types.Warning{}

	// Extract provider options
	provOpts, extra, err := extractVideoProviderOptions(opts.ProviderOptions)
	if err != nil {
		return nil, err
	}

	// Check for unsupported options and add warnings
	warnings = append(warnings, m.checkUnsupportedOptions(opts, provOpts)...)

	// Determine if this is video editing or generation
	isEdit := provOpts.VideoURL != nil && *provOpts.VideoURL != ""

	// Build request body
	body := m.buildRequestBody(opts, provOpts, extra, isEdit)

	// Determine endpoint
	endpoint := "/v1/videos/generations"
	if isEdit {
		endpoint = "/v1/videos/edits"
	}

	// Submit video generation/edit request
	var createResp xaiVideoCreateResponse
	if err := m.provider.client.PostJSON(ctx, endpoint, body, &createResp); err != nil {
		return nil, m.handleError(err)
	}

	if createResp.RequestID == "" {
		return nil, providererrors.NewProviderError("xai", 0, "",
			fmt.Sprintf("No request_id returned from xAI API. Response: %+v", createResp), nil)
	}

	// Poll for completion
	pollInterval := 5 * time.Second
	if provOpts.PollIntervalMs != nil && *provOpts.PollIntervalMs > 0 {
		pollInterval = time.Duration(*provOpts.PollIntervalMs) * time.Millisecond
	}

	pollTimeout := 600 * time.Second
	if provOpts.PollTimeoutMs != nil && *provOpts.PollTimeoutMs > 0 {
		pollTimeout = time.Duration(*provOpts.PollTimeoutMs) * time.Millisecond
	}

	// Use polling utility
	pollOpts := polling.PollOptions{
		PollIntervalMs: int(pollInterval.Milliseconds()),
		PollTimeoutMs:  int(pollTimeout.Milliseconds()),
	}

	statusChecker := func(ctx context.Context) (*polling.JobResult, error) {
		var status xaiVideoStatusResponse
		statusPath := fmt.Sprintf("/v1/videos/%s", createResp.RequestID)

		if err := m.provider.client.GetJSON(ctx, statusPath, &status); err != nil {
			return nil, m.handleError(err)
		}

		// Check if done
		if status.Status == "done" || (status.Status == "" && status.Video != nil && status.Video.URL != "") {
			// Check for moderation rejection: respect_moderation == false means blocked.
			if status.Video != nil && status.Video.RespectModeration != nil && !*status.Video.RespectModeration {
				return nil, &ModerationError{
					Code:    "",
					Message: "Video generation was blocked due to a content policy violation.",
				}
			}

			if status.Video == nil || status.Video.URL == "" {
				return nil, providererrors.NewProviderError("xai", 0, "",
					"Video generation completed but no video URL was returned", nil)
			}
			return &polling.JobResult{
				Status:    polling.JobStatusCompleted,
				OutputURL: status.Video.URL,
				Metadata: map[string]interface{}{
					"video": status.Video,
					"model": status.Model,
					"usage": status.Usage,
				},
			}, nil
		}

		// Check if expired
		if status.Status == "expired" {
			return &polling.JobResult{
				Status: polling.JobStatusFailed,
				Error:  "Video generation request expired",
			}, nil
		}

		// Still pending
		return &polling.JobResult{
			Status: polling.JobStatusProcessing,
		}, nil
	}

	jobResult, err := polling.PollForCompletion(ctx, statusChecker, pollOpts)

	if err != nil {
		return nil, err
	}

	// Extract video data from metadata
	videoData := jobResult.Metadata["video"].(*xaiVideoData)

	// Build xai-scoped metadata.
	xaiMeta := map[string]interface{}{
		"requestId": createResp.RequestID,
		"videoUrl":  videoData.URL,
	}
	if videoData.Duration != nil {
		xaiMeta["duration"] = *videoData.Duration
	}
	// Cost is in the top-level usage object, not inside the video object.
	if usageData, ok := jobResult.Metadata["usage"].(*xaiVideoUsage); ok && usageData != nil {
		if usageData.CostInUsdTicks != nil {
			xaiMeta["costInUsdTicks"] = *usageData.CostInUsdTicks
		}
	}

	// Build response
	resp := &provider.VideoModelV3Response{
		Videos: []provider.VideoModelV3VideoData{
			{
				Type:      "url",
				URL:       videoData.URL,
				MediaType: "video/mp4",
			},
		},
		Warnings: warnings,
		ProviderMetadata: map[string]interface{}{
			"xai": xaiMeta,
		},
		Response: provider.VideoModelV3ResponseInfo{
			Timestamp: time.Now(),
			ModelID:   m.modelID,
			Headers:   map[string]string{},
		},
	}

	return resp, nil
}

// buildRequestBody constructs the API request body
func (m *VideoModel) buildRequestBody(opts *provider.VideoModelV3CallOptions, provOpts *XAIVideoProviderOptions, extra map[string]interface{}, isEdit bool) map[string]interface{} {
	body := map[string]interface{}{
		"model":  m.modelID,
		"prompt": opts.Prompt,
	}

	// Add duration (not for edits)
	if !isEdit && opts.Duration != nil {
		body["duration"] = *opts.Duration
	}

	// Add aspect ratio (not for edits)
	if !isEdit && opts.AspectRatio != "" {
		body["aspect_ratio"] = opts.AspectRatio
	}

	// Add resolution (not for edits)
	if !isEdit && provOpts.Resolution != nil {
		body["resolution"] = *provOpts.Resolution
	} else if !isEdit && opts.Resolution != "" {
		// Map standard resolution to XAI format
		mapped := mapResolution(opts.Resolution)
		if mapped != "" {
			body["resolution"] = mapped
		}
	}

	// Video editing: add source video URL
	if isEdit && provOpts.VideoURL != nil {
		body["video"] = map[string]interface{}{
			"url": *provOpts.VideoURL,
		}
	}

	// Image-to-video: add source image
	if opts.Image != nil {
		body["image"] = m.convertImageToXAIFormat(opts.Image)
	}

	// Passthrough any extra provider options not handled above.
	for k, v := range extra {
		body[k] = v
	}

	return body
}

// convertImageToXAIFormat converts VideoModelV3File to XAI image format
func (m *VideoModel) convertImageToXAIFormat(img *provider.VideoModelV3File) map[string]interface{} {
	if img.Type == "url" {
		return map[string]interface{}{
			"url": img.URL,
		}
	}

	// Convert binary data to base64 data URL
	base64Data := base64.StdEncoding.EncodeToString(img.Data)
	mediaType := img.MediaType
	if mediaType == "" {
		mediaType = "image/png"
	}
	dataURL := fmt.Sprintf("data:%s;base64,%s", mediaType, base64Data)

	return map[string]interface{}{
		"url": dataURL,
	}
}

// checkUnsupportedOptions checks for unsupported options and generates warnings
func (m *VideoModel) checkUnsupportedOptions(opts *provider.VideoModelV3CallOptions, provOpts *XAIVideoProviderOptions) []types.Warning {
	warnings := []types.Warning{}
	isEdit := provOpts.VideoURL != nil && *provOpts.VideoURL != ""

	if opts.FPS != nil {
		warnings = append(warnings, types.Warning{
			Type:    "unsupported-option",
			Message: "xAI video models do not support custom FPS",
		})
	}

	if opts.Seed != nil {
		warnings = append(warnings, types.Warning{
			Type:    "unsupported-option",
			Message: "xAI video models do not support seed",
		})
	}

	if opts.N > 1 {
		warnings = append(warnings, types.Warning{
			Type:    "unsupported-option",
			Message: "xAI video models do not support generating multiple videos per call. Only 1 video will be generated.",
		})
	}

	if isEdit && opts.Duration != nil {
		warnings = append(warnings, types.Warning{
			Type:    "unsupported-option",
			Message: "xAI video editing does not support custom duration",
		})
	}

	if isEdit && opts.AspectRatio != "" {
		warnings = append(warnings, types.Warning{
			Type:    "unsupported-option",
			Message: "xAI video editing does not support custom aspect ratio",
		})
	}

	if isEdit && (provOpts.Resolution != nil || opts.Resolution != "") {
		warnings = append(warnings, types.Warning{
			Type:    "unsupported-option",
			Message: "xAI video editing does not support custom resolution",
		})
	}

	return warnings
}

// mapResolution maps standard resolution strings to XAI format
func mapResolution(resolution string) string {
	resolutionMap := map[string]string{
		"1280x720": "720p",
		"854x480":  "480p",
		"640x480":  "480p",
	}

	if mapped, ok := resolutionMap[resolution]; ok {
		return mapped
	}

	return ""
}

// extractVideoProviderOptions extracts XAI-specific provider options and any
// unrecognized keys (which are passed through to the API request body).
func extractVideoProviderOptions(opts map[string]interface{}) (*XAIVideoProviderOptions, map[string]interface{}, error) {
	if opts == nil {
		return &XAIVideoProviderOptions{}, nil, nil
	}

	xaiRaw, ok := opts["xai"]
	if !ok {
		return &XAIVideoProviderOptions{}, nil, nil
	}

	// Convert to JSON and back to struct
	jsonData, err := json.Marshal(xaiRaw)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to marshal provider options: %w", err)
	}

	var provOpts XAIVideoProviderOptions
	if err := json.Unmarshal(jsonData, &provOpts); err != nil {
		return nil, nil, fmt.Errorf("failed to unmarshal provider options: %w", err)
	}

	// Collect any unrecognized keys for passthrough to the API.
	known := map[string]bool{
		"pollIntervalMs": true, "pollTimeoutMs": true,
		"resolution": true, "videoUrl": true,
	}
	extra := make(map[string]interface{})
	if rawMap, ok := xaiRaw.(map[string]interface{}); ok {
		for k, v := range rawMap {
			if !known[k] {
				extra[k] = v
			}
		}
	}

	return &provOpts, extra, nil
}

// handleError converts provider errors
func (m *VideoModel) handleError(err error) error {
	if provErr, ok := err.(*providererrors.ProviderError); ok {
		return provErr
	}
	return providererrors.NewProviderError("xai", 0, "", err.Error(), err)
}

// xaiVideoCreateResponse represents the video creation API response
type xaiVideoCreateResponse struct {
	RequestID string `json:"request_id"`
}

// xaiVideoStatusResponse represents the video status API response
type xaiVideoStatusResponse struct {
	Status string         `json:"status"`
	Video  *xaiVideoData  `json:"video,omitempty"`
	Model  string         `json:"model,omitempty"`
	Usage  *xaiVideoUsage `json:"usage,omitempty"`
}

// xaiVideoUsage holds top-level usage data from the video status response.
type xaiVideoUsage struct {
	CostInUsdTicks *int64 `json:"cost_in_usd_ticks,omitempty"`
}

// xaiVideoData represents video data in the status response
type xaiVideoData struct {
	URL               string   `json:"url"`
	Duration          *float64 `json:"duration,omitempty"`
	RespectModeration *bool    `json:"respect_moderation,omitempty"`
}
