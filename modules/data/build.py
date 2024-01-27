from torch.utils.data import DataLoader
from .datasets import BaseDataSet, tripletInfo_collate_fn, image_collate_fn
from .samplers import TripletInfoNceSampler, ImageSampler

def build_data(cfg, test_on=None):
	if test_on is None:
		train_set = BaseDataSet(cfg, 'TRAIN')
		valid_set = BaseDataSet(cfg, 'VALID')
		train_loader = DataLoader(
			train_set,
			collate_fn=tripletInfo_collate_fn,
			batch_sampler=TripletInfoNceSampler(cfg),
			num_workers=cfg.DATA.NUM_WORKERS,
			pin_memory=True
		)
		valid_query_loader = DataLoader(
			valid_set,
			collate_fn=image_collate_fn,
			batch_sampler=ImageSampler(cfg, cfg.DATA.GROUNDTRUTH.QUERY.VALID),
			num_workers=cfg.DATA.NUM_WORKERS,
			pin_memory=True
		)

		valid_candidate_loader = DataLoader(
			valid_set,
			collate_fn=image_collate_fn,
			batch_sampler=ImageSampler(cfg, cfg.DATA.GROUNDTRUTH.CANDIDATE.VALID),
			num_workers=cfg.DATA.NUM_WORKERS,
			pin_memory=True
		)

		return train_loader, valid_query_loader, valid_candidate_loader
	else:
		test_set = BaseDataSet(cfg, test_on)

		test_candidate_loader = DataLoader(
			test_set,
			collate_fn=image_collate_fn,
			batch_sampler=ImageSampler(cfg, cfg.DATA.GROUNDTRUTH.CANDIDATE[test_on]),
			num_workers=cfg.DATA.NUM_WORKERS,
			pin_memory=True
		)

		return test_candidate_loader